import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig, MoEConfig


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.swiglu = SwiGLU(config.dim, config.ffn_dim)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.swiglu(x))


class MoE(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.moe_config: MoEConfig = config.moe
        self.num_experts = self.moe_config.num_experts
        self.top_k = self.moe_config.top_k
        self.capacity_factor = self.moe_config.capacity_factor
        self.aux_loss_coef = self.moe_config.aux_loss_coef
        self.z_loss_coef = self.moe_config.z_loss_coef

        self.gate = nn.Linear(config.dim, self.num_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLU(config.dim, config.ffn_dim) for _ in range(self.num_experts)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        *leading_dims, d_model = x.shape
        x_flat = x.view(-1, d_model)
        num_tokens = x_flat.shape[0]

        logits = self.gate(x_flat)
        gates = F.softmax(logits.float(), dim=-1).type_as(logits)
        top_k_gates, top_k_indices = torch.topk(gates, self.top_k, dim=-1)

        top_k_gates = top_k_gates / (top_k_gates.sum(dim=-1, keepdim=True) + 1e-9)
        top_k_gates = top_k_gates.to(x.dtype)

        final_output = torch.zeros_like(x_flat)

        for expert_idx, expert in enumerate(self.experts):
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)
            selected_indices = torch.where(expert_mask)[0]
            if selected_indices.numel() > 0:
                expert_inputs = x_flat[selected_indices]
                expert_outputs = expert(expert_inputs)
                expert_gates = top_k_gates[selected_indices, :]
                expert_weight = expert_gates[top_k_indices[selected_indices] == expert_idx]
                weighted_output = expert_outputs * expert_weight.unsqueeze(-1)
                final_output[selected_indices] = final_output[selected_indices] + weighted_output

        final_output = final_output.view(*leading_dims, d_model)

        aux_loss = self._compute_aux_loss(gates, top_k_indices)
        z_loss = self._compute_z_loss(logits)

        return final_output, aux_loss + z_loss

    def _compute_aux_loss(self, gates: torch.Tensor, top_k_indices: torch.Tensor) -> torch.Tensor:
        num_tokens = gates.shape[0]
        num_experts = self.num_experts

        probs = F.one_hot(top_k_indices, num_classes=num_experts).float().mean(dim=0)
        probs = probs.mean(dim=0)
        frac = probs

        gate_weights = gates.mean(dim=0)
        aux_loss = num_experts * (frac * gate_weights).sum()
        return self.aux_loss_coef * aux_loss

    def _compute_z_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return self.z_loss_coef * (logits.float().logsumexp(dim=-1) ** 2).mean()
