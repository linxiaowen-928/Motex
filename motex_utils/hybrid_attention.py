"""Motex 混合注意力（Dense + Sparse）中的可复用组件。

注：HybridAttentionBlock 因 v2.1 / v2.2 的 top_k 等配置不同，保留在各版本 notebook 内。
"""

import torch
from torch import nn

from .transformer import (DotProductFlashAttention, apply_rotary_pos_emb,
                          precompute_rotary_emb, repeat_kv, transpose_output,
                          transpose_qkv)


class DenseAttention(nn.Module):
    """标准软注意力：独立 Q/K/V 投影 + RoPE + GQA + KV Cache + Flash Attention"""

    def __init__(self, query_size, key_size, value_size, num_hiddens, num_heads, num_kv_heads,
                 dropout, max_seq_len, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = num_hiddens // num_heads

        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_kv_heads * self.head_dim, bias=bias)
        self.W_v = nn.Linear(value_size, num_kv_heads * self.head_dim, bias=bias)

        self.flash_attn = DotProductFlashAttention(dropout)
        self.cos, self.sin = precompute_rotary_emb(max_seq_len, self.head_dim)

    def forward(self, queries, keys, values, valid_lens, state, i, query_offset):
        Q = self.W_q(queries)
        K = self.W_k(keys)
        V = self.W_v(values)

        Q = transpose_qkv(Q, self.num_heads)
        K = transpose_qkv(K, self.num_kv_heads)
        V = transpose_qkv(V, self.num_kv_heads)

        Q = apply_rotary_pos_emb(Q, self.cos, self.sin, offset=query_offset)
        K = apply_rotary_pos_emb(K, self.cos, self.sin, offset=query_offset)

        if not self.training and state is not None and state[0] is not None and state[0][i] is not None:
            cache_k, cache_v = state[0][i]
            K = torch.cat((cache_k, K), dim=1)
            V = torch.cat((cache_v, V), dim=1)
        if not self.training and state is not None and state[0] is not None:
            state[0][i] = (K, V)

        n_rep = self.num_heads // self.num_kv_heads
        K = repeat_kv(K, n_rep, self.num_heads, self.num_kv_heads)
        V = repeat_kv(V, n_rep, self.num_heads, self.num_kv_heads)

        # GQA：把 (batch,) 的 valid_lens 按 num_heads 重复为 (batch*num_heads,)，
        # 与 attention.py 的 GQARopeMultiHeadAttentionKVCache 保持一致。
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)
        out = self.flash_attn(Q, K, V, valid_lens)
        out = transpose_output(out, self.num_heads)
        return out, state


class SparseAttentionRouter(nn.Module):
    """纯 Top-K 选择器，不拥有任何投影参数"""

    def __init__(self, top_k):
        super().__init__()
        self.top_k = top_k

    def forward(self, scores):
        topk_scores, topk_indices = torch.topk(scores, k=min(self.top_k, scores.size(-1)), dim=-1)
        topk_weights = torch.softmax(topk_scores, dim=-1)
        return topk_indices, topk_weights


class SparseAttention(nn.Module):
    """硬注意力通路：独立 Q/K/V 投影 + RoPE + GQA + 独立 Router + 收集加权"""

    def __init__(self, query_size, key_size, value_size, num_hiddens, num_heads, num_kv_heads,
                 top_k, max_seq_len, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = num_hiddens // num_heads
        self.scale = self.head_dim ** 0.5

        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_kv_heads * self.head_dim, bias=bias)
        self.W_v = nn.Linear(value_size, num_kv_heads * self.head_dim, bias=bias)

        self.router = SparseAttentionRouter(top_k)
        self.cos, self.sin = precompute_rotary_emb(max_seq_len, self.head_dim)

    def forward(self, queries, keys, values, query_offset, causal_mask=None, state=None, i=0):
        Q = self.W_q(queries)
        K = self.W_k(keys)
        V = self.W_v(values)

        Q = transpose_qkv(Q, self.num_heads)
        K = transpose_qkv(K, self.num_kv_heads)
        V = transpose_qkv(V, self.num_kv_heads)

        Q = apply_rotary_pos_emb(Q, self.cos, self.sin, offset=query_offset)
        K = apply_rotary_pos_emb(K, self.cos, self.sin, offset=query_offset)

        # [问题] 硬（稀疏）通路原先完全不拼历史缓存：解码时 keys=当前 token，只能‘注意到自己’，
        #       与训练时能看到整个因果上下文的行为不一致（KV-Cache 一致性测试 max_diff≈0.037）。
        # [解决] 与 DenseAttention 一致地拼接/更新缓存；硬通路投影与软通路不共享，故用独立缓存槽 state[1][i]。
        if not self.training and state is not None and state[0] is not None:
            if len(state) < 2 or state[1] is None:
                state.append([None] * len(state[0]))
            slot = state[1]
            if slot[i] is not None:
                cache_k, cache_v = slot[i]
                K = torch.cat((cache_k, K), dim=1)
                V = torch.cat((cache_v, V), dim=1)
            slot[i] = (K, V)

        n_rep = self.num_heads // self.num_kv_heads
        K = repeat_kv(K, n_rep, self.num_heads, self.num_kv_heads)
        V = repeat_kv(V, n_rep, self.num_heads, self.num_kv_heads)

        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        if causal_mask is not None:
            scores = scores + causal_mask.to(scores.device)

        topk_indices, topk_weights = self.router(scores)

        # [修复] V 的真实键长 SL = S(当前) + 历史缓存；不再假设 SL==S（加缓存后解码时二者不等）。
        #       用 SL 计算行内偏移，用 Sq=topk_indices.size(1) 记录当前 query 数。
        BH, SL, D = V.shape
        K_dim = topk_indices.size(-1)
        Sq = topk_indices.size(1)          # 当前 query 数
        offset = torch.arange(BH, device=V.device).view(BH, 1, 1) * SL
        flat_indices = (topk_indices + offset).reshape(-1)
        V_flat = V.reshape(-1, D)
        V_topk = V_flat[flat_indices].reshape(BH, Sq, K_dim, D)

        out = (topk_weights.unsqueeze(-1) * V_topk).sum(dim=2)
        out = transpose_output(out, self.num_heads)
        return out
