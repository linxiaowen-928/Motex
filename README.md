# Motex

从零手写、自底向上演进的 **Decoder-only 语言模型**系列（原学习项目代号 "Moss"，此处更名 Motex）。

本仓库只包含**模型本体**（架构 + 共享模型组件 + 各版本 notebook），
**不含**：数据加载/数据集、训练运行脚本、测试/验证脚本、过程文档与断点产物 ——
这些都整理在 `dev/` 目录（已 git 忽略，不随仓库分发）。

## 目录结构

```
Motex/
├── gpt/                      # GPT 系列（模型定义 notebook）
│   ├── GPT_v1.ipynb          # RoPE + 标准多头注意力 + FFN
│   └── GPT_v2.ipynb          # + KV-Cache、AMP、断点训练
├── motex/                    # Motex 系列
│   ├── Motex_v1.ipynb        # RMSNorm + GQA + KV-Cache（Pre-Norm）
│   ├── Motex_v2.ipynb        # + SwiGLU + MoE
│   ├── Motex_v2_1.ipynb      # + HybridAttention（Dense/Sparse）
│   ├── Motex_v2_2.ipynb      # SwiGLU + MoE + HybridAttention（完整版）
│   └── Motex_v3.ipynb        # 内置因果掩码 + 权重绑定（干净版）
├── motex_utils/              # ★ 各版本复用/依赖的模型组件
│   ├── transformer.py        # RoPE / 掩码 softmax / GQA 辅助 / RMSNorm / FFN 等基础件
│   ├── attention.py          # GQA(RoPE+KV-Cache) 注意力
│   ├── moe.py                # SwiGLU / MoE 路由器与专家 FFN
│   ├── hybrid_attention.py   # DenseAttention / SparseAttention(Router)
│   ├── motex_v3.py           # MotexV3（内置因果掩码 / 权重绑定 / 统一接口 / 生成）
│   ├── bpe.py                # 轻量纯 Python BPE 分词器（模型侧工具）
│   └── training.py           # 共享训练/评估/推理/断点工具（各 notebook 复用）
├── requirements.txt
└── README.md
```

## 模型统一接口

各 Motex 版本模型对外统一为：

```python
forward(tokens, valid_lens=None, state=None) -> (logits, state, aux_loss)
```

- `tokens`: `(batch, seq_len)` 的 token 索引；
- `valid_lens`: 可选（v3 内置因果掩码，不依赖它保证因果性）；
- `state`: KV-Cache 状态（逐层 K/V；增量解码时传入）；
- 返回 `logits: (batch, seq_len, vocab)`、更新后的 `state`、以及 `aux_loss`
  （无 MoE 的版本恒为 0.0）。

因此各版本可共用 `motex_utils.training` 里的训练 / 预测 / 断点函数。

## 版本演进与关键机制

| 版本 | 机制 |
|---|---|
| GPT v1 → v2 | RoPE + 多头注意力 + FFN → + KV-Cache（推理避免重复投影） |
| Motex v1 | RMSNorm + GQA + Pre-Norm（显著降低 KV 缓存显存） |
| Motex v2 | + SwiGLU + MoE 稀疏激活 |
| Motex v2_1 / v2_2 | + Dense/Sparse 混合注意力（top_k 16 / 8） |
| Motex v3 | **内置因果掩码**（不依赖外部 valid_lens）+ 权重绑定 + 干净化实现 |

## 依赖

- `torch`、`numpy`、`matplotlib`、`tqdm`
- `d2l`（仅 Animator/Timer/try_gpu，notebook 用）
- `deepseek-tokenizer`（提供 `ds_token` 分词与词表；字符/BPE 训练请在 `dev/` 管线中自行构建）

## 模型用法

```python
import torch
from motex_utils.motex_v3 import MotexV3

net = MotexV3(vocab_size=4096, d_model=256, num_layers=4,
              num_heads=4, num_kv_heads=2, ffn_hidden=512,
              dropout=0.1, max_seq_len=128)

ids = torch.randint(5, 4096, (2, 128))
logits, state, aux = net(ids, None, None)   # 训练前向
```

> 数据加载为仓库外接口：训练前请在对应 notebook 的「数据加载（placeholder）」处
> 接入返回 `(tokens, labels, valid_lens)` 的迭代器；真实数据/训练/验证/测试脚本在 `dev/`。

## dev/（不入库）

`dev/` 存有本次开发过程所用的训练管线、语料构建、多变体对比、自定义训练实验、
冒烟/扩展验证、过程报告与聊天室接入等，均被 `.gitignore` 忽略，不随本仓库分发。
