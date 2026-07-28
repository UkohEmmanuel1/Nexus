import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .rope import apply_rotary_emb


class Attention(nn.Module):
    def __init__(self, config: ModelConfig, layer_id: int = 0):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = config.n_rep
        self.layer_id = layer_id

        self.sliding_window = config.long_context.sliding_window_size if config.long_context.enabled else 0

        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)

        self.attn_dropout = nn.Dropout(config.attention_dropout)

        self.cache_k: torch.Tensor = None
        self.cache_v: torch.Tensor = None

    def init_kv_cache(
        self, batch_size: int, max_seq_len: int, device: torch.device, dtype: torch.dtype
    ):
        self.cache_k = torch.zeros(
            batch_size, max_seq_len, self.n_kv_heads, self.head_dim, device=device, dtype=dtype
        )
        self.cache_v = torch.zeros(
            batch_size, max_seq_len, self.n_kv_heads, self.head_dim, device=device, dtype=dtype
        )

    def reset_kv_cache(self):
        self.cache_k = None
        self.cache_v = None

    def _sliding_window_mask(self, seq_len: int, start_pos: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.sliding_window <= 0 or seq_len <= self.sliding_window:
            return None
        mask = torch.full((seq_len, seq_len + start_pos), float("-inf"), device=device, dtype=dtype)
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype), diagonal=1)
        mask[:, :start_pos] = 0.0
        mask[:, start_pos:] = causal_mask

        window_start = max(0, seq_len + start_pos - self.sliding_window)
        mask[:, :window_start] = float("-inf")
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        start_pos: int = 0,
        mask: torch.Tensor = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis[start_pos : start_pos + seq_len])

        if use_cache and self.cache_k is not None:
            self.cache_k[:bsz, start_pos : start_pos + seq_len] = xk
            self.cache_v[:bsz, start_pos : start_pos + seq_len] = xv
            keys = self.cache_k[:bsz, : start_pos + seq_len]
            values = self.cache_v[:bsz, : start_pos + seq_len]
        else:
            keys = xk
            values = xv

        if self.n_rep > 1:
            keys = keys.repeat_interleave(self.n_rep, dim=2)
            values = values.repeat_interleave(self.n_rep, dim=2)

        if mask is None and self.sliding_window > 0 and start_pos == 0:
            mask = self._sliding_window_mask(seq_len, start_pos, x.device, x.dtype)

        if self.config.use_flash_attn:
            scale = self.head_dim ** -0.5
            attn_output = F.scaled_dot_product_attention(
                xq.transpose(1, 2),
                keys.transpose(1, 2),
                values.transpose(1, 2),
                attn_mask=mask,
                dropout_p=self.config.attention_dropout if self.training else 0.0,
                scale=scale,
            )
            attn_output = attn_output.transpose(1, 2).contiguous()
        else:
            scale = self.head_dim ** -0.5
            scores = torch.matmul(xq.transpose(1, 2), keys.transpose(1, 2).transpose(-2, -1)) * scale
            if mask is not None:
                scores = scores + mask
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            attn_output = torch.matmul(scores, values.transpose(1, 2))
            attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, seq_len, -1)
        return self.wo(attn_output)
