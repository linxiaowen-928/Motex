"""Motex 稀疏 MoE 与 SwiGLU 模块。"""

import torch
from torch import nn
from torch.nn import functional as F


def swiglu_activation(x, dim=-1):
    """纯 SwiGLU 激活函数，不包含任何线性层"""
    a, b = x.chunk(2, dim=dim)
    return F.silu(a) * b


class SwiGLUMLP(nn.Module):
    def __init__(self, ffn_num_inputs: int, ffn_num_hiddens: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(ffn_num_inputs, ffn_num_hiddens, bias=False)
        self.up_proj = nn.Linear(ffn_num_inputs, ffn_num_hiddens, bias=False)
        self.down_proj = nn.Linear(ffn_num_hiddens, ffn_num_inputs, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))


class MoeTopKRouter(nn.Module):
    def __init__(self, num_inputs, num_experts, top_k, default_aux_coef=1e-2, z_loss_coef=1e-3):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.default_aux_coef = default_aux_coef
        self.z_loss_coef = z_loss_coef
        self.dense = nn.Linear(num_inputs, num_experts, bias=False)

    def forward(self, X):
        B, D = X.shape
        logits = self.dense(X)
        probs = torch.softmax(logits, dim=-1)

        topk_weights, topk_indices = torch.topk(probs, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        if self.training:
            importance = probs.mean(dim=0)
            topk_onehot = F.one_hot(topk_indices, num_classes=self.num_experts).float()
            load = (topk_onehot * topk_weights.unsqueeze(-1)).sum(dim=0).sum(dim=0) / B
            aux_loss = self.num_experts * torch.sum(importance * load)
            z_loss = logits.pow(2).mean()
            aux_loss = aux_loss + self.z_loss_coef * z_loss
        else:
            aux_loss = torch.tensor(0.0, device=X.device)

        return topk_weights, topk_indices, aux_loss


class MoeExpertFFN(nn.Module):
    def __init__(self, ffn_num_inputs, ffn_num_hiddens, ffn_num_outs, dropout, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ffn = SwiGLUMLP(ffn_num_inputs=ffn_num_inputs, ffn_num_hiddens=ffn_num_hiddens,
                             dropout=dropout)

    def forward(self, X):
        return self.ffn(X)


class MoEFeedForward(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_experts, top_k, dropout,
                 use_shared_expert=True, aux_loss_coef=1e-2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_shared_expert = use_shared_expert
        self.aux_loss_coef = aux_loss_coef

        self.router = MoeTopKRouter(d_model, num_experts, top_k)

        self.experts = nn.ModuleList([
            MoeExpertFFN(d_model, ffn_hidden, d_model, dropout)
            for _ in range(num_experts)
        ])

        if use_shared_expert:
            self.shared_expert = MoeExpertFFN(d_model, ffn_hidden, d_model, dropout)
        else:
            self.shared_expert = None

    def forward(self, x):
        """x: (batch, seq_len, d_model) -> (output, aux_loss)"""
        B, S, D = x.shape
        flat_x = x.view(-1, D)
        topk_weights, topk_indices, aux_loss = self.router(flat_x)

        T = flat_x.size(0)
        K = self.top_k
        E = self.num_experts

        expert_ids = topk_indices.view(-1)
        weights = topk_weights.view(-1, 1)
        token_idx = torch.arange(T, device=flat_x.device).unsqueeze(1).expand(T, K).reshape(-1)

        sorted_expert_ids, sort_indices = torch.sort(expert_ids)
        sorted_token_idx = token_idx[sort_indices]
        sorted_weights = weights[sort_indices]

        bin_ids, counts = torch.unique_consecutive(sorted_expert_ids, return_counts=True)
        offsets = torch.cumsum(counts, dim=0) - counts

        output = torch.zeros_like(flat_x)
        for i in range(len(bin_ids)):
            expert_id = bin_ids[i].item()
            start = offsets[i].item()
            end = start + counts[i].item()
            cur_tokens = sorted_token_idx[start:end]
            cur_weights = sorted_weights[start:end]
            expert_input = flat_x[cur_tokens]
            expert_out = self.experts[expert_id](expert_input)
            weighted_out = expert_out * cur_weights
            output.scatter_add_(0, cur_tokens.unsqueeze(1).expand(-1, D), weighted_out)

        if self.shared_expert is not None:
            shared_out = self.shared_expert(flat_x)
            output = output + shared_out

        output = output.view(B, S, D)
        return output, aux_loss
