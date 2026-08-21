"""Motex_v3：自清洁的 Decoder-only 语言模型（夜间实验版）。

相对 v1/v2/v2_1/v2_2 的改进点：
1. 内置因果注意力掩码：不再依赖外部 valid_lens 的传法来保证因果性，
   训练/预填充/增量生成在结构上自洽（旧的注意力在 valid_lens=None 时会把训练变双向）。
2. 标准组合：Pre-Norm(RMSNorm) + GQA(RoPE + KV-Cache) + SwiGLU FFN + 权重绑定。
3. 统一返回 (logits, state, aux_loss=0.0)，兼容 motex_utils.training 的训练/预测函数。
4. 纯自包含：只依赖 torch 与 motex_utils.transformer / moe 的基础组件。

注意：本模块使用「字符级」小词表训练时效果最佳（若用大 BPE 词表请自行调整 LM 头大小）。
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .transformer import (RMSNorm, apply_rotary_pos_emb, causal_bias,
                          precompute_rotary_emb, repeat_kv, transpose_output,
                          transpose_qkv)
from .moe import SwiGLUMLP


class GQARopeCausalAttention(nn.Module):
    """GQA + RoPE + KV-Cache，且内置因果掩码的注意力。

    支持可选改进：【RoPE 外推 (NTK-Aware 缩放)】
    - 默认（rope_scaling=None）：使用标准 RoPE 基频 base=100000（原始指数频率）。
      问题：RoPE 是"绝对位置 + 旋转"，超出训练时的最大位置后，位置向量已旋转过多圈，
      注意力会迅速失效（外推失败）——即"训练到 256 就只能在 256 内用"。
    - 改进（rope_scaling={'type':'ntk','alpha':k}）：NTK-Aware 缩放，把基频放大
      base' = base * alpha ** (head_dim/(head_dim-2))，
      相当于把频率谱向低频方向拉伸，使更长位置仍在合理相位内 → 模型可以外推到
      约 alpha 倍于训练长度的上下文（无需重训）。是"改位置编码"的推理侧增强。
    - 用法：__init__ 里传 rope_scaling；A/B 对比时分别开/关该参数即可。
    """

    def __init__(self, d_model, num_heads, num_kv_heads, dropout, max_seq_len, bias=False,
                 rope_base=100000.0, rope_scaling=None, use_sdp=False, qk_norm=True):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** 0.5
        self.use_sdp = use_sdp        # 改进④：出前向走 F.scaled_dot_product_attention(FlashAttention 内核)
        self.qk_norm = qk_norm        # 改进⑤：softmax 前对 Q/K 做 RMSNorm（低精度训练稳定）
        # 注：qk_norm 默认 True——A/B（dev/IMPROVEMENT_LOG.md 改进项5）显示它域内验证最优、
        # 外推 CE 从 11.6→6.1 的大幅提升且零成本，故作为 motex_v3 默认；SDPA(use_sdp) 在本规模
        # (d512/8L/单批) 无速度收益，保留为可选（大模型/长上下文/大批量场景启用）。

        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=bias)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)
        base_ = rope_base
        if rope_scaling and rope_scaling.get('type') == 'ntk':
            alpha = float(rope_scaling.get('alpha', 1.0))
            base_ = rope_base * (alpha ** (self.head_dim / (self.head_dim - 2.0)))
        self.rope_base = base_
        self.cos, self.sin = precompute_rotary_emb(max_seq_len, self.head_dim, base=base_)

    def forward(self, x, state, i):
        # x: (B, S, d_model) -> (out: B, S, d_model, state)
        B, S, _ = x.shape

        q = transpose_qkv(self.W_q(x), self.num_heads)          # (B*H, S, hd)
        k = transpose_qkv(self.W_k(x), self.num_kv_heads)
        v = transpose_qkv(self.W_v(x), self.num_kv_heads)

        # 是否有缓存，决定当前 token 的绝对位置（RoPE offset）
        offset = 0
        cache_k = cache_v = None
        if (not self.training) and state is not None and state[0] is not None and state[0][i] is not None:
            cache_k, cache_v = state[0][i]
            offset = cache_k.shape[1]

        q = apply_rotary_pos_emb(q, self.cos, self.sin, offset=offset)
        k = apply_rotary_pos_emb(k, self.cos, self.sin, offset=offset)

        if offset:
            k = torch.cat([cache_k, k], dim=1)
            v = torch.cat([cache_v, v], dim=1)
        if (not self.training) and state is not None and state[0] is not None:
            state[0][i] = (k, v)

        n_rep = self.num_heads // self.num_kv_heads
        k = repeat_kv(k, n_rep, self.num_heads, self.num_kv_heads)
        v = repeat_kv(v, n_rep, self.num_heads, self.num_kv_heads)

        # [改进⑤] QK-Norm：softmax 前对每个头 Q/K 做 RMSNorm（教学版用无参数；工业版带可学习 scale）。
        # 作用：把注意力分数尺度钉在稳定范围，低精度（bf16/fp16）训练显著更稳、不易爆数。
        if self.qk_norm:
            q = F.rms_norm(q, (self.head_dim,))
            k = F.rms_norm(k, (self.head_dim,))

        # 内置因果掩码：历史列全放行，当前 S 列施加上三角 -inf（训练 offset=0 即纯因果；解码 S=1 全 0）
        total = S + offset
        mask = torch.zeros((S, total), device=q.device, dtype=q.dtype)
        if S > 1:
            tri = torch.triu(torch.full((S, S), float('-inf'), device=q.device,
                                        dtype=q.dtype), diagonal=1)
            mask[:, offset:] = tri

        if self.use_sdp:
            # [改进④] 真·FlashAttention：F.scaled_dot_product_attention 在 CUDA 上会自动选择
            # FlashAttention / Memory-Efficient 内核（不物化超大 attention 分数矩阵 → 提速省显存）。
            # 注：仓库里以前的 “DotProductFlashAttention” 只是名字叫 Flash 的普通 softmax 注意力。
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                                 dropout_p=self.dropout.p if self.training else 0.0)
        else:
            scores = torch.bmm(q, k.transpose(1, 2)) / self.scale
            scores = scores + mask
            w = F.softmax(scores, dim=-1)
            w = self.dropout(w)
            out = torch.bmm(w, v)
        out = transpose_output(out, self.num_heads)
        return self.W_o(out), state


class MLAAttention(nn.Module):
    """MLA（Multi-head Latent Attention，DeepSeek-V2 风格，教学简化版）——可选改进②。

    动机 / 对比：
    - GQA 会把每层的 K、V（shape: num_kv_heads*head_dim 各一份）直接缓存，
      每 token 每层缓存量 = 2 * num_kv_heads * head_dim（默认 kv=2, hd=64 → 256 个 float）。
    - MLA 的做法：先用 W_dkv 把输入压进一个**低维潜在向量 c**（d_c 很小，默认 32），
      KV 缓存**只存 c**；解码时再用 W_upK/W_upV 从 c 恢复出 K/V。
      于是每 token 每层缓存量 = d_c（32 个 float），约为 GQA 的 1/8 → 长上下文/多层的
      KV 显存大幅下降，代价是解码时多一步低秩恢复计算（K/V 是低秩重构，非满秩投影）。

    本实现是教学简化版（差异说明）：
    1) DeepSeek-V2 对 Q 也用低秩压缩（吸收进权重）、K/V 恢复后做分头 RoPE；
       这里 Q 保持 GQA 同款全量投影，K 恢复后再统一施加绝对位置 RoPE（等价且更直观）。
    2) 缓存只存 latent，解码时整段恢复 K/V 再 RoPE（简单、显存最优；计算略增，
       大模型可用“每步只恢复新 token”的优化）。
    - A/B 对比实验见 dev/IMPROVEMENT_LOG.md「改进项 2」。
    """

    def __init__(self, d_model, num_heads, num_kv_heads, dropout, max_seq_len, bias=False,
                 latent_dim=32, rope_base=100000.0, rope_scaling=None,
                 use_sdp=False, qk_norm=True):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.latent_dim = latent_dim
        self.scale = self.head_dim ** 0.5
        self.use_sdp = use_sdp        # 改进④
        self.qk_norm = qk_norm        # 改进⑤（默认开，理由见 GQARopeCausalAttention）

        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_dkv = nn.Linear(d_model, latent_dim, bias=bias)          # 输入 → 潜在 c
        self.W_upK = nn.Linear(latent_dim, num_kv_heads * self.head_dim, bias=bias)  # c → K
        self.W_upV = nn.Linear(latent_dim, num_kv_heads * self.head_dim, bias=bias)  # c → V
        self.W_o = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)
        base_ = rope_base
        if rope_scaling and rope_scaling.get('type') == 'ntk':
            alpha = float(rope_scaling.get('alpha', 1.0))
            base_ = rope_base * (alpha ** (self.head_dim / (self.head_dim - 2.0)))
        self.cos, self.sin = precompute_rotary_emb(max_seq_len, self.head_dim, base=base_)

    def kv_cache_bytes_per_token(self):
        """每 token 每层 KV 缓存字节数（GQA 对比见 IMPROVEMENT_LOG 改进项2）"""
        return 4 * self.latent_dim

    def forward(self, x, state, i):
        # x: (B, S, d_model) -> (out: B, S, d_model, state)
        B, S, _ = x.shape
        q = transpose_qkv(self.W_q(x), self.num_heads)          # (B*H, S, hd)

        offset = 0
        cache_c = None
        if (not self.training) and state is not None and state[0] is not None and state[0][i] is not None:
            cache_c = state[0][i]                               # 只缓存 latent：(B, off, d_c)
            offset = cache_c.shape[1]
        q = apply_rotary_pos_emb(q, self.cos, self.sin, offset=offset)

        c = self.W_dkv(x)                                       # (B, S, d_c)
        c_all = torch.cat([cache_c, c], dim=1) if cache_c is not None else c
        if (not self.training) and state is not None and state[0] is not None:
            state[0][i] = c_all                                 # 更新缓存（仍只是 latent）

        k = transpose_qkv(self.W_upK(c_all), self.num_kv_heads) # (B*kv, off+S, hd)
        v = transpose_qkv(self.W_upV(c_all), self.num_kv_heads)
        k = apply_rotary_pos_emb(k, self.cos, self.sin)         # 全列绝对位置（offset=0）

        n_rep = self.num_heads // self.num_kv_heads
        k = repeat_kv(k, n_rep, self.num_heads, self.num_kv_heads)
        v = repeat_kv(v, n_rep, self.num_heads, self.num_kv_heads)

        # [改进⑤] QK-Norm（MLAV 同款）：见 GQARopeCausalAttention 注释
        if self.qk_norm:
            q = F.rms_norm(q, (self.head_dim,))
            k = F.rms_norm(k, (self.head_dim,))

        # 内置因果掩码
        M = causal_bias(q.shape[1], k.shape[1], q.device, q.dtype)

        if self.use_sdp:
            # [改进④] 真·FlashAttention 内核（见 GQARopeCausalAttention 注释）
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=M,
                                                 dropout_p=self.dropout.p if self.training else 0.0)
        else:
            scores = torch.bmm(q, k.transpose(1, 2)) / self.scale   # (B*H, S, S+offset)
            scores = scores + M
            w = F.softmax(scores, dim=-1)
            w = self.dropout(w)
            out = torch.bmm(w, v)
        out = transpose_output(out, self.num_heads)
        return self.W_o(out), state


class MotexV3Block(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, ffn_hidden, dropout, i, max_seq_len,
                 rope_base=100000.0, rope_scaling=None, attn='gqa', mla_latent_dim=32,
                 use_sdp=False, qk_norm=True):
        super().__init__()
        self.i = i
        self.norm1 = RMSNorm(d_model)
        if attn == 'mla':
            # 改进② MLA：只缓存低维 latent，KV 显存≈1/8（对比见 dev/IMPROVEMENT_LOG.md）
            self.attn = MLAAttention(d_model, num_heads, num_kv_heads, dropout, max_seq_len,
                                     rope_base=rope_base, rope_scaling=rope_scaling,
                                     latent_dim=mla_latent_dim,
                                     use_sdp=use_sdp, qk_norm=qk_norm)
        else:
            self.attn = GQARopeCausalAttention(d_model, num_heads, num_kv_heads, dropout,
                                               max_seq_len, rope_base=rope_base,
                                               rope_scaling=rope_scaling,
                                               use_sdp=use_sdp, qk_norm=qk_norm)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLUMLP(d_model, ffn_hidden, dropout)

    def forward(self, x, state=None, valid_lens=None):
        h = self.norm1(x)
        attn_out, state = self.attn(h, state, self.i)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, state, 0.0   # 非 MoE，aux_loss=0


class MotexV3Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, num_kv_heads,
                 ffn_hidden, dropout, max_seq_len, rope_base=100000.0, rope_scaling=None,
                 attn='gqa', mla_latent_dim=32, use_sdp=False, qk_norm=False):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.blks = nn.ModuleList([
            MotexV3Block(d_model, num_heads, num_kv_heads, ffn_hidden, dropout, i, max_seq_len,
                         rope_base=rope_base, rope_scaling=rope_scaling,
                         attn=attn, mla_latent_dim=mla_latent_dim,
                         use_sdp=use_sdp, qk_norm=qk_norm)
            for i in range(num_layers)
        ])

    def forward(self, tokens, valid_lens=None, state=None):
        x = self.token_emb(tokens)
        if state is not None and state[0] is None:
            state[0] = [None] * self.num_layers
        aux = 0.0
        for blk in self.blks:
            x, state, a = blk(x, state, valid_lens)
            aux += a
        return x, state, aux / len(self.blks)


class MotexV3(nn.Module):
    """统一接口：forward(tokens, valid_lens=None, state=None) -> (logits, state, aux_loss)。"""

    def __init__(self, vocab_size, d_model=512, num_layers=8, num_heads=8, num_kv_heads=2,
                 ffn_hidden=1024, dropout=0.1, max_seq_len=256,
                 rope_base=100000.0, rope_scaling=None, attn='gqa', mla_latent_dim=32,
                 use_sdp=False, qk_norm=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.decoder = MotexV3Decoder(vocab_size, d_model, num_layers, num_heads,
                                      num_kv_heads, ffn_hidden, dropout, max_seq_len,
                                      rope_base=rope_base, rope_scaling=rope_scaling,
                                      attn=attn, mla_latent_dim=mla_latent_dim,
                                      use_sdp=use_sdp, qk_norm=qk_norm)
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # 权重绑定：embedding 与 LM 头共享，显著降低大词表开销、加速收敛
        self.lm_head.weight = self.decoder.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, tokens, valid_lens=None, state=None):
        x, state, aux = self.decoder(tokens, valid_lens, state)
        logits = self.lm_head(self.norm(x))
        return logits, state, aux


@torch.no_grad()
def generate(net, tokenizer, prompt, max_new_tokens, device, temperature=0.8, top_k=40,
             repetition_penalty=1.15):
    """字符级自回归生成（带 KV-Cache）。
    - temperature/top_k 控制随机性；
    - repetition_penalty>1 对已生成 token 的 logits 施加惩罚，抑制小模型常见的循环重复；
    - 输出自动跳过 <pad>/<bos>/<eos> 等特殊 token。
    tokenizer 需提供 encode(text)->list[int] 与 stoi/itos。"""
    net.eval()
    ids = tokenizer.encode(prompt)
    special = {tokenizer.stoi[s] for s in ('<pad>', '<bos>', '<eos>') if s in tokenizer.stoi}
    num_layers = net.decoder.num_layers
    state = [[None] * num_layers, [None] * num_layers]
    input_ids = torch.tensor([ids], device=device)

    logits, state, _ = net(input_ids, valid_lens=None, state=state)

    def pick(logts, seen):
        logts = logts.clone()
        if repetition_penalty and repetition_penalty != 1.0 and seen:
            for t in seen:
                logts[t] /= repetition_penalty
        logts = logts / temperature
        probs = F.softmax(logts, dim=-1)
        topk = torch.topk(probs, min(top_k, probs.numel()))
        return topk.indices[torch.multinomial(topk.values, 1)].item()

    seen = []
    next_id = pick(logits[0, -1, :], seen)
    gen = []
    for _ in range(max_new_tokens):
        gen.append(next_id)
        seen.append(next_id)
        one = torch.tensor([[next_id]], device=device)
        logits, state, _ = net(one, valid_lens=None, state=state)
        next_id = pick(logits[0, -1, :], seen)

    text = ''.join(tokenizer.itos[i] for i in gen if i not in special)
    return prompt + text, gen
