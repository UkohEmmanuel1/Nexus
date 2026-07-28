import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from .attention import Attention
from .config import ModelConfig
from .feedforward import FeedForward, MoE
from .rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, config: ModelConfig):
        super().__init__()
        self.layer_id = layer_id
        self.config = config

        self.attention = Attention(config, layer_id=layer_id)
        self.feed_forward = MoE(config) if config.moe.enabled else FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.gradient_checkpointing = config.gradient_checkpointing

    def _forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        start_pos: int = 0,
        mask: torch.Tensor = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x
        x = self.attention_norm(x)
        x = self.attention(x, freqs_cis, start_pos, mask, use_cache)
        h = residual + x

        residual = h
        h = self.ffn_norm(h)
        if isinstance(self.feed_forward, MoE):
            h, aux_loss = self.feed_forward(h)
        else:
            h = self.feed_forward(h)
            aux_loss = torch.tensor(0.0, device=h.device)

        out = residual + h
        return out, aux_loss

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        start_pos: int = 0,
        mask: torch.Tensor = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.gradient_checkpointing and self.training:
            return checkpoint.checkpoint(
                self._forward,
                x,
                freqs_cis,
                start_pos,
                mask,
                use_cache,
                use_reentrant=False,
            )
        return self._forward(x, freqs_cis, start_pos, mask, use_cache)
