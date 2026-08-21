"""Motex 基础组件：多头注意力相关的小工具与常用模块。"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def transpose_qkv(X, num_heads):
    """(batch, len, num_hiddens) -> (batch*num_heads, len, head_dim)"""
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """逆转 transpose_qkv 的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)
    mask = torch.arange(maxlen, dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


def masked_softmax(X, valid_lens):
    """带有效长度掩码的 softmax；valid_lens 为 None 时退化为普通 softmax"""
    if valid_lens is None:
        return F.softmax(X, dim=-1)
    shape = X.shape
    if valid_lens.dim() == 1:
        valid_lens = torch.repeat_interleave(valid_lens, shape[1])
    else:
        valid_lens = valid_lens.reshape(-1)
    X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-float('inf'))
    return F.softmax(X.reshape(shape), dim=-1)


def causal_bias(seq_len, total, device, dtype=torch.float32):
    """构造 (seq_len, total) 的加法因果掩码（内置，不依赖外部 valid_lens）：
    - 前 total-seq_len 列 = 历史缓存列，全部放行（0）
    - 后 seq_len 列 = 当前 token 列，上三角置 -inf（禁止看未来）
    训练/整段前向时 total==seq_len，即纯因果方阵；增量解码时 seq_len==1，只看历史+自己。
    """
    mask = torch.zeros((seq_len, total), device=device, dtype=dtype)
    if seq_len > 1:
        tri = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=dtype),
                         diagonal=1)
        mask[:, total - seq_len:] = tri
    return mask


def precompute_rotary_emb(max_seq_len, d, base=100000):
    theta = 1.0 / (base ** (torch.arange(0, d, 2, dtype=torch.float) / d))
    positions = torch.arange(max_seq_len, dtype=torch.float)
    angles = positions.unsqueeze(1) * theta.unsqueeze(0)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin


def apply_rotary_pos_emb(x, cos, sin, offset=0):
    seq_len = x.shape[-2]
    cos = cos[offset:offset + seq_len, :].to(x.device)
    sin = sin[offset:offset + seq_len, :].to(x.device)
    while cos.dim() < x.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    half_d = x.shape[-1] // 2
    x_left = x[..., :half_d]
    x_right = x[..., half_d:]
    x_rotated_left = x_left * cos - x_right * sin
    x_rotated_right = x_left * sin + x_right * cos
    return torch.cat([x_rotated_left, x_rotated_right], dim=-1)


def repeat_kv(x, n_rep, num_heads, num_kv_heads):
    batch_times_kv, seq_len, head_dim = x.shape
    batch = batch_times_kv // num_kv_heads
    x = x.reshape(batch, num_kv_heads, seq_len, head_dim)
    x = x.unsqueeze(2).expand(-1, -1, n_rep, -1, -1)
    x = x.reshape(batch, num_heads, seq_len, head_dim)
    return x.reshape(batch * num_heads, seq_len, head_dim)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x.float()
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x / rms * self.weight).to(x.dtype)


class DotProductAttention(nn.Module):
    """缩放点积注意力（支持加法掩码，如内置因果掩码）"""

    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None, mask=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        if mask is not None:
            scores = scores + mask
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


class DotProductFlashAttention(nn.Module):
    """缩放点积注意力（支持加法掩码，如内置因果掩码）"""

    def __init__(self, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None, mask=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        if mask is not None:
            scores = scores + mask
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


class AddNorm(nn.Module):
    """残差连接 + LayerNorm"""

    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class PositionWiseFFN(nn.Module):
    """逐位置前馈网络（GELU 版）"""

    def __init__(self, ffn_num_inputs, ffn_num_hiddens, ffn_num_outs, dropout=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense1 = nn.Linear(ffn_num_inputs, ffn_num_hiddens)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outs)

    def forward(self, X):
        return self.dense2(self.dropout(self.activation(self.dense1(X))))
