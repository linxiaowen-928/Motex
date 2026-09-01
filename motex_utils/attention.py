"""Motex 共用注意力模块：GQA + RoPE + KV-Cache 多头注意力。

包含：
- GQARopeMultiHeadAttentionKVCache：v1/v2 使用的 GQA(RoPE+KV-Cache) 注意力（掩码依赖 valid_lens）
- GQARopeCausalAttention：内置因果掩码 + 可选改进（QK-Norm / RoPE 外推 / SDPA），v3 组装用
- MLAAttention：低秩 latent KV（KV 缓存 1/8），v4 组装用
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .transformer import (DotProductFlashAttention, apply_rotary_pos_emb,
                          causal_bias, precompute_rotary_emb, repeat_kv,
                          transpose_output, transpose_qkv)


class GQARopeMultiHeadAttentionKVCache(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads,
                 dropout, max_seq_len, bias=False, num_kv_heads=None, **kwargs):
        super().__init__()
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        if num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        self.num_heads = num_heads
        self.head_dim = num_hiddens // num_heads

        self.attention = DotProductFlashAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.W_v = nn.Linear(value_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

        self.cos, self.sin = self.precompute_rotary_emb(max_seq_len, self.head_dim)

    def forward(self, queries, keys, values, valid_lens, state, i):
        queries = self.W_q(queries)
        keys = self.W_k(keys)
        values = self.W_v(values)

        if not self.training and state is not None and state[0] is not None and state[0][i] is not None:
            cache_k, cache_v = state[0][i]
            query_offset = cache_k.shape[1]
        else:
            query_offset = 0

        queries = self.transpose_qkv(queries, self.num_heads)
        keys = self.transpose_qkv(keys, self.num_kv_heads)
        values = self.transpose_qkv(values, self.num_kv_heads)

        queries = self.apply_rotary_pos_emb(queries, self.cos, self.sin, offset=query_offset)
        keys = self.apply_rotary_pos_emb(keys, self.cos, self.sin, offset=query_offset)

        if not self.training and state is not None and state[0] is not None and state[0][i] is not None:
            keys = torch.cat((cache_k, keys), dim=1)
            values = torch.cat((cache_v, values), dim=1)

        if not self.training and state is not None and state[0] is not None:
            state[0][i] = (keys, values)

        n_rep = self.num_heads // self.num_kv_heads
        keys = self.repeat_kv(keys, n_rep)
        values = self.repeat_kv(values, n_rep)

        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

        # [修复/回迁] 内置因果掩码（不依赖外部 valid_lens 的传法）：
        #   训练/整段前向 total(=S+query_offset)=S 时即为纯因果方阵；增量解码时只看历史+自己。
        M = causal_bias(queries.shape[1], keys.shape[1], queries.device, queries.dtype)
        output = self.attention(queries, keys, values, valid_lens, mask=M)
        output_concat = self.transpose_output(output, self.num_heads)
        return self.W_o(output_concat), state

    def precompute_rotary_emb(self, max_seq_len, d, base=100000):
        theta = 1.0 / (base ** (torch.arange(0, d, 2, dtype=torch.float) / d))
        positions = torch.arange(max_seq_len, dtype=torch.float)
        angles = positions.unsqueeze(1) * theta.unsqueeze(0)
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        return cos, sin

    def apply_rotary_pos_emb(self, x, cos, sin, offset=0):
        seq_len = x.shape[-2]
        d = x.shape[-1]
        half_d = d // 2
        cos = cos[offset:offset + seq_len, :].to(x.device)
        sin = sin[offset:offset + seq_len, :].to(x.device)
        while cos.dim() < x.dim():
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        x_left = x[..., :half_d]
        x_right = x[..., half_d:]
        x_rotated_left = x_left * cos - x_right * sin
        x_rotated_right = x_left * sin + x_right * cos
        return torch.cat([x_rotated_left, x_rotated_right], dim=-1)

    def transpose_qkv(self, X, num_heads):
        X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
        X = X.permute(0, 2, 1, 3)
        return X.reshape(-1, X.shape[2], X.shape[3])

    def transpose_output(self, X, num_heads):
        """逆转 transpose_qkv 函数的操作"""
        X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)

    def repeat_kv(self, x, n_rep):
        """x: (batch * num_kv_heads, seq_len, head_dim)"""
        batch_times_kv_heads, seq_len, head_dim = x.shape
        batch = batch_times_kv_heads // self.num_kv_heads
        x = x.reshape(batch, self.num_kv_heads, seq_len, head_dim)
        x = x.unsqueeze(2)
        x = x.expand(-1, -1, n_rep, -1, -1)
        x = x.reshape(batch, self.num_kv_heads * n_rep, seq_len, head_dim)
        x = x.reshape(batch * self.num_heads, seq_len, head_dim)
        return x


class GQARopeCausalAttention(nn.Module):
    """GQA + RoPE + KV-Cache，且内置因果掩码的注意力（v3 组装用）。

    内置因果掩码：不依赖外部 valid_lens 的传法保证因果性（训练/预填充/增量生成结构上自洽）。

    可选改进（默认值即 v3 默认）：
    - qk_norm=True：softmax 前对每个头 Q/K 做 RMSNorm（教学版无参数，工业版带可学习 scale）。
      作用：把注意力分数尺度钉在稳定范围，低精度（bf16/fp16）训练显著更稳；A/B 实验显示
      域内验证最优 + 外推 CE 11.6→6.1，零成本，故默认开启。
    - rope_scaling={'type':'ntk','alpha':k}（默认 None）：NTK-Aware RoPE 外推，
      把旋转基频按 base*alpha^(d/(d-2)) 拉伸，可外推约 alpha 倍训练长度（本规模实验为负结果，默认关）。
    - use_sdp=True（默认 False）：走 F.scaled_dot_product_attention（CUDA 上自动选真 FlashAttention
      内核，免物化大分数矩阵）；本规模(d512/8L/单批)无提速，大模型/长上下文/大批量再启用。
    """

    def __init__(self, d_model, num_heads, num_kv_heads, dropout, max_seq_len, bias=False,
                 rope_base=100000.0, rope_scaling=None, use_sdp=False, qk_norm=True):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** 0.5
        self.use_sdp = use_sdp
        self.qk_norm = qk_norm

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

        if self.qk_norm:
            q = F.rms_norm(q, (self.head_dim,))
            k = F.rms_norm(k, (self.head_dim,))

        if self.use_sdp:
            # 4D 视图触发 flash attention（3D 输入会回退 math 路径，慢 ~9 倍）
            B_h, S_q, hd = q.shape
            B = B_h // self.num_heads
            q4 = q.view(B, self.num_heads, S_q, hd)
            k4 = k.view(B, self.num_heads, S_q, hd)
            v4 = v.view(B, self.num_heads, S_q, hd)
            if offset == 0:
                # 训练（纯因果）：融合 kernel + is_causal，免去掩码构建
                out = F.scaled_dot_product_attention(
                    q4, k4, v4, is_causal=True,
                    dropout_p=self.dropout.p if self.training else 0.0)
            else:
                # 增量解码（offset>0，罕见）：构建因果掩码后走融合 kernel
                total = S + offset
                mask = torch.zeros((S, total), device=q.device, dtype=q.dtype)
                if S > 1:
                    tri = torch.triu(torch.full((S, S), float('-inf'), device=q.device,
                                                dtype=q.dtype), diagonal=1)
                    mask[:, offset:] = tri
                out = F.scaled_dot_product_attention(q4, k4, v4, attn_mask=mask,
                                                     dropout_p=self.dropout.p if self.training else 0.0)
            out = out.reshape(B_h, S_q, hd)
        else:
            # 内置因果掩码：历史列全放行，当前 S 列施加上三角 -inf（训练 offset=0 即纯因果；解码 S=1 全 0）
            total = S + offset
            mask = torch.zeros((S, total), device=q.device, dtype=q.dtype)
            if S > 1:
                tri = torch.triu(torch.full((S, S), float('-inf'), device=q.device,
                                            dtype=q.dtype), diagonal=1)
                mask[:, offset:] = tri
            scores = torch.bmm(q, k.transpose(1, 2)) / self.scale
            scores = scores + mask
            w = F.softmax(scores, dim=-1)
            w = self.dropout(w)
            out = torch.bmm(w, v)
        out = transpose_output(out, self.num_heads)
        return self.W_o(out), state


class MLAAttention(nn.Module):
    """MLA（Multi-head Latent Attention，DeepSeek-V2 风格，教学简化版）——v4 组装用。

    动机 / 对比：
    - GQA 会把每层的 K、V（shape: num_kv_heads*head_dim 各一份）直接缓存，
      每 token 每层缓存量 = 2 * num_kv_heads * head_dim（默认 kv=2, hd=64 → 256 个 float）。
    - MLA 的做法：先用 W_dkv 把输入压进一个低维潜在向量 c（d_c 很小，默认 32），
      KV 缓存只存 c；解码时再用 W_upK/W_upV 从 c 恢复出 K/V。
      于是每 token 每层缓存量 = d_c（32 个 float），约为 GQA 的 1/8 → 长上下文/多层的
      KV 显存大幅下降，代价是解码时多一步低秩恢复计算。

    教学简化（与 DeepSeek-V2 的差异）：
    1) DeepSeek-V2 对 Q 也用低秩压缩（吸收进权重）、K/V 恢复后做分头 RoPE；
       这里 Q 保持 GQA 同款全量投影，K 恢复后再统一施加绝对位置 RoPE（等价且更直观）。
    2) 缓存只存 latent，解码时整段恢复 K/V 再 RoPE（简单、显存最优；计算略增）。
    经 A/B 对比实验验证：KV 缓存 1/8、域内略优、外推更优。
    """

    def __init__(self, d_model, num_heads, num_kv_heads, dropout, max_seq_len, bias=False,
                 latent_dim=32, rope_base=100000.0, rope_scaling=None,
                 use_sdp=False, qk_norm=True, decoupled_rope=False):
        """decoupled_rope（简化变体，2026-08-24）：
        位置编码只施加于 head_dim 后半（位置段），前半（内容段）不旋转——
        DeepSeek MLA decoupled-RoPE 的轻量近似（不拆 latent 缓存，零结构变化）、默认 False 兼容。"""
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.latent_dim = latent_dim
        self.scale = self.head_dim ** 0.5
        self.use_sdp = use_sdp
        self.qk_norm = qk_norm   # 默认开，理由见 GQARopeCausalAttention 注释
        self.decoupled_rope = decoupled_rope
        self._rope_half = self.head_dim // 2 if decoupled_rope else 0

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
        """每 token 每层 KV 缓存字节数（MLA 与 GQA 缓存量对比见类 docstring）"""
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
        if self.decoupled_rope:
            r = self._rope_half                       # 位置段维数（r 个 dim = r/2 对）
            pairs = r // 2
            q = torch.cat([q[..., :r],
                           apply_rotary_pos_emb(q[..., r:], self.cos[:, :pairs], self.sin[:, :pairs],
                                                offset=offset)], dim=-1)
        else:
            q = apply_rotary_pos_emb(q, self.cos, self.sin, offset=offset)

        c = self.W_dkv(x)                                       # (B, S, d_c)
        c_all = torch.cat([cache_c, c], dim=1) if cache_c is not None else c
        if (not self.training) and state is not None and state[0] is not None:
            state[0][i] = c_all                                 # 更新缓存（仍只是 latent）

        k = transpose_qkv(self.W_upK(c_all), self.num_kv_heads) # (B*kv, off+S, hd)
        v = transpose_qkv(self.W_upV(c_all), self.num_kv_heads)
        if self.decoupled_rope:
            r = self._rope_half
            pairs = r // 2
            k = torch.cat([k[..., :r],
                           apply_rotary_pos_emb(k[..., r:], self.cos[:, :pairs], self.sin[:, :pairs])], dim=-1)
        else:
            k = apply_rotary_pos_emb(k, self.cos, self.sin)     # 全列绝对位置（offset=0）

        n_rep = self.num_heads // self.num_kv_heads
        k = repeat_kv(k, n_rep, self.num_heads, self.num_kv_heads)
        v = repeat_kv(v, n_rep, self.num_heads, self.num_kv_heads)

        if self.qk_norm:
            q = F.rms_norm(q, (self.head_dim,))
            k = F.rms_norm(k, (self.head_dim,))

        if self.use_sdp:
            # 4D 视图触发 flash attention（3D 输入回退 math 路径，慢 ~9 倍）
            B_h, S_q, hd = q.shape
            B = B_h // self.num_heads
            causal = (S_q == k.shape[1])
            out = F.scaled_dot_product_attention(
                q.view(B, self.num_heads, S_q, hd),
                k.view(B, self.num_heads, S_q, hd),
                v.view(B, self.num_heads, S_q, hd),
                is_causal=True if causal else False,
                attn_mask=None if causal else causal_bias(S_q, k.shape[1], q.device, q.dtype),
                dropout_p=self.dropout.p if self.training else 0.0)
            out = out.reshape(B_h, S_q, hd)
        else:
            M = causal_bias(q.shape[1], k.shape[1], q.device, q.dtype)
            scores = torch.bmm(q, k.transpose(1, 2)) / self.scale   # (B*H, S, S+offset)
            scores = scores + M
            w = F.softmax(scores, dim=-1)
            w = self.dropout(w)
            out = torch.bmm(w, v)
        out = transpose_output(out, self.num_heads)
        return self.W_o(out), state
