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

from .transformer import (RMSNorm, apply_rotary_pos_emb, precompute_rotary_emb,
                          repeat_kv, transpose_output, transpose_qkv)
from .moe import SwiGLUMLP


class GQARopeCausalAttention(nn.Module):
    """GQA + RoPE + KV-Cache，且内置因果掩码的注意力。"""

    def __init__(self, d_model, num_heads, num_kv_heads, dropout, max_seq_len, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** 0.5

        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=bias)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.cos, self.sin = precompute_rotary_emb(max_seq_len, self.head_dim)

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

        scores = torch.bmm(q, k.transpose(1, 2)) / self.scale   # (B*H, S, S+offset)

        # 内置因果掩码：历史列全放行，当前 S 列施加上三角 -inf（训练 offset=0 即纯因果；解码 S=1 全 0）
        total = S + offset
        mask = torch.zeros((S, total), device=scores.device, dtype=scores.dtype)
        if S > 1:
            tri = torch.triu(torch.full((S, S), float('-inf'), device=scores.device,
                                        dtype=scores.dtype), diagonal=1)
            mask[:, offset:] = tri
        scores = scores + mask

        w = F.softmax(scores, dim=-1)
        w = self.dropout(w)
        out = torch.bmm(w, v)
        out = transpose_output(out, self.num_heads)
        return self.W_o(out), state


class MotexV3Block(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, ffn_hidden, dropout, i, max_seq_len):
        super().__init__()
        self.i = i
        self.norm1 = RMSNorm(d_model)
        self.attn = GQARopeCausalAttention(d_model, num_heads, num_kv_heads, dropout, max_seq_len)
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
                 ffn_hidden, dropout, max_seq_len):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.blks = nn.ModuleList([
            MotexV3Block(d_model, num_heads, num_kv_heads, ffn_hidden, dropout, i, max_seq_len)
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
                 ffn_hidden=1024, dropout=0.1, max_seq_len=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.decoder = MotexV3Decoder(vocab_size, d_model, num_layers, num_heads,
                                      num_kv_heads, ffn_hidden, dropout, max_seq_len)
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
