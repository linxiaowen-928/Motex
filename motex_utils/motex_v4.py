"""Motex_v4：已验证改进的默认组合（模型本体）。

v4 = v3（内置因果掩码 / 权重绑定 / 统一接口）+ 已验证的组合默认：
  1. MLA（Multi-head Latent Attention，attn='mla'）：KV 缓存降到 GQA 的 1/8，
     A/B（docs/IMPROVEMENT_LOG.md 改进项2）显示域内略优、外推更优。
  2. QK-Norm（qk_norm=True）：softmax 前对 Q/K 做 RMSNorm，
     A/B（改进项5）显示域内最优 + 外推 CE 11.6→6.1，默认开启。
  3. bf16/AMP 训练（改进项3，2× 吞吐、0 NaN）：训练侧推荐（见 dev/ 训练脚本）。
其余保持可选：rope_scaling（改进项1，实验为负结果，默认关）、use_sdp（改进项4，
本规模无提速，大模型/长上下文场景再开）。

实现上 v4 直接复用 motex_v3.MotexV3（全部机制在 v3 中已实现并注释），
MotexV4 只是把『已验证组合』固化为默认参数：attn='mla'、qk_norm=True。
"""

from .motex_v3 import MotexV3


class MotexV4(MotexV3):
    """Motex V4：默认 MLA + QK-Norm 组合（其余参数同 MotexV3）。"""

    def __init__(self, vocab_size, d_model=512, num_layers=8, num_heads=8, num_kv_heads=2,
                 ffn_hidden=1024, dropout=0.1, max_seq_len=256,
                 rope_base=100000.0, rope_scaling=None, attn='mla', mla_latent_dim=32,
                 use_sdp=False, qk_norm=True):
        # 与 v3 唯一区别：attn 默认 'mla'（v3 默认 'gqa'）；qk_norm 默认 True（v3 同）。
        super().__init__(vocab_size=vocab_size, d_model=d_model, num_layers=num_layers,
                         num_heads=num_heads, num_kv_heads=num_kv_heads,
                         ffn_hidden=ffn_hidden, dropout=dropout, max_seq_len=max_seq_len,
                         rope_base=rope_base, rope_scaling=rope_scaling, attn=attn,
                         mla_latent_dim=mla_latent_dim, use_sdp=use_sdp, qk_norm=qk_norm)