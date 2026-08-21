"""Motex 共用注意力模块：GQA + RoPE + KV-Cache 多头注意力。"""

import torch
from torch import nn

from .transformer import DotProductFlashAttention, causal_bias


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
