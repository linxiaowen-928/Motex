"""Mamba 混合架构实验件（plan.md 第二篇：方案 A/B/C 的地基）。

本文件提供：
- MambaBlock：S4 风格简化 SSM 层（固定/可学 delta 的离散化 + conv + 门控）——
  数值稳定的第一步实现，后续可升级 selective scan（按 Mamba 论文）；
- build_hybrid(...)：按方案 C（堆叠 6M+6T）/ A（层间交替）/ B（层内并行预留）组装。

设计对齐 plan.md：8G 显存、与 MotexV3 同接口 (logits, state, aux)。
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn

from motex_utils.transformer import RMSNorm


class S4SSM(nn.Module):
    """连续系统 (A,B,C,Δ) 的零阶保持离散化 + 递推：
    h_{t} = Ah_{t-1} + Bx_t ；y_t = C h_t
    A 用对角线复表示（S4 风格），Δ 可学（简化：每维标量），简化实现只取实部（数值稳）。
    """
    def __init__(self, d_state=16, d_model=768):
        super().__init__()
        self.d_state = d_state
        # 可学离散化速率（每维度一个标量 Δ）
        self.log_dt = nn.Parameter(torch.log(torch.full((d_model,), 0.01)))
        # 对角线 A（实部对数参数化保证 <0）
        self.A_log = nn.Parameter(torch.randn(d_model, d_state))
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.05)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.05)

    def forward(self, x, state=None):
        # x: (B, S, d)
        B, S, d = x.shape
        dt = torch.exp(self.log_dt).unsqueeze(-1)                    # (d, 1)
        A = -torch.exp(self.A_log)                                   # (d, ds)
        A_bar = A * dt                                               # (d, ds)
        B_bar = self.B * dt                                          # (d, ds)
        C = self.C                                                   # (d, ds)
        if state is None:
            h = torch.zeros(B, d, self.d_state, device=x.device)
        else:
            h = state
        outs = []
        for t in range(S):
            h = h * A_bar + x[:, t, :].unsqueeze(-1) * B_bar         # (B,d,ds)
            outs.append(torch.einsum('bkd,kd->bk', h, C))
        y = torch.stack(outs, dim=1)                                # (B,S,d)
        return y, h


class MambaBlock(nn.Module):
    """简化 Mamba 层：x → norm → SSM（+可选 conv 门控）→ 残差。
    注：无 selective scan（固定 Δ 对角 A），用于架构对比的混合验证；后续可升级。"""
    def __init__(self, d_model, d_state=16, dropout=0.1):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.ssm = S4SSM(d_state, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, state=None, i=0):
        h = self.norm(x)
        g, x2 = self.in_proj(h).chunk(2, dim=-1)          # 门控
        y, h_new = self.ssm(x2, state)
        y = self.out_proj(y * torch.sigmoid(g))
        return x + self.drop(y), h_new


def build_hybrid(vocab_size, d_model=768, num_layers=12, num_heads=12, num_kv_heads=4,
                 ffn_hidden=2048, dropout=0.1, max_seq_len=768, mamba_first=6,
                 attn='gqa', use_sdp=True, qk_norm=True, d_state=16):
    """方案 C（堆叠）：前 mamba_first 层 Mamba，后 (num_layers-mamba_first) 层 Transformer。
    返回与 Motex 同接口的模块：forward(tokens, valid_lens, state) -> (logits, state, aux)。
    说明：为对比实验的轻量组装（复用 motex.model.MotexDecoder 的 Transformer 块）。"""
    from motex.model import MotexDecoder, MotexV3
    base = MotexV3(vocab_size=vocab_size, d_model=d_model, num_layers=num_layers,
                   num_heads=num_heads, num_kv_heads=num_kv_heads, ffn_hidden=ffn_hidden,
                   dropout=dropout, max_seq_len=max_seq_len, attn=attn, use_sdp=use_sdp,
                   qk_norm=qk_norm)
    # 用 Mamba 层替换前 mamba_first 个 Transformer 块（保持接口 state[0]/state[1]）
    mambas = nn.ModuleList([MambaBlock(d_model, d_state, dropout) for _ in range(mamba_first)])
    base.mambas = mambas
    base.mamba_first = mamba_first

    orig_forward = base.decoder.forward

    def hybrid_decoder(tokens, valid_lens=None, state=None):
        x = base.decoder.token_emb(tokens)
        if state is not None and state[0] is None:
            state[0] = [None] * base.decoder.num_layers
            state[1] = [None] * base.decoder.num_layers
        for i in range(mamba_first):
            x, h = base.mambas[i](x, state[1][i] if state is not None else None)
            if state is not None:
                state[1][i] = h
        aux = 0.0
        for j, blk in enumerate(base.decoder.blks[mamba_first:]):
            x, state, a = blk(x, state, j + mamba_first)
            aux += a
        return x, state, aux / max(1, len(base.decoder.blks) - mamba_first)

    base.decoder.forward = hybrid_decoder
    return base


if __name__ == '__main__':
    # CPU smoke：堆叠混合（2 Mamba + 2 T）可前向/反向
    import torch
    net = build_hybrid(vocab_size=100, d_model=64, num_layers=4, num_heads=4, num_kv_heads=2,
                       ffn_hidden=128, max_seq_len=128, mamba_first=2, d_state=8)
    x = torch.randint(0, 100, (2, 32))
    lg, st, aux = net(x, None, None)
    print('hybrid forward:', tuple(lg.shape), 'aux=', float(aux))
    lg.float().mean().backward()
    print('backward OK | 参数量 %.2fM' % (sum(p.numel() for p in net.parameters()) / 1e6))
    # 增量解码态（Mamba 层 state 传递）
    st = [[None] * 4, [None] * 4]
    for _ in range(3):
        lg, st, _ = net(torch.randint(0, 100, (1, 1)), None, st)
    print('incremental OK:', tuple(lg.shape))