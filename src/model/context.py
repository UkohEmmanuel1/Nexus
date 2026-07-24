import torch
from typing import List, Optional, Tuple

from .config import ModelConfig


class HierarchicalMemory:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.chunk_size = config.long_context.chunk_size
        self.compress_ratio = config.long_context.compress_ratio
        self.memory_tokens = config.long_context.memory_tokens
        self.memories: List[torch.Tensor] = []
        self.segment_ids: List[int] = []

    def clear(self):
        self.memories = []
        self.segment_ids = []

    def add_segment(self, hidden_states: torch.Tensor, segment_id: int):
        self.memories.append(hidden_states)
        self.segment_ids.append(segment_id)
        max_memories = self.config.max_seq_len // self.chunk_size
        while len(self.memories) > max_memories:
            self.memories.pop(0)
            self.segment_ids.pop(0)

    def get_memory_tokens(self, n_layers: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if not self.memories:
            return None
        pooled = []
        for mem in self.memories[-self.memory_tokens:]:
            if mem.dim() == 3:
                pooled.append(mem.mean(dim=1, keepdim=True))
        if not pooled:
            return None
        return torch.cat(pooled, dim=1).to(device=device, dtype=dtype)


def create_long_context_mask(
    seq_len: int,
    chunk_size: int,
    n_memory_tokens: int,
    has_summary: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    total_len = seq_len
    if has_summary:
        total_len += n_memory_tokens
    mask = torch.full((total_len, total_len), float("-inf"), device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)

    if has_summary:
        mask[:n_memory_tokens, :] = 0.0
        mask[n_memory_tokens:, :n_memory_tokens] = 0.0

    chunk_boundaries = list(range(n_memory_tokens if has_summary else 0, total_len, chunk_size))
    for i, start in enumerate(chunk_boundaries):
        end = min(start + chunk_size, total_len)
        if i > 0:
            prev_start = chunk_boundaries[i - 1]
            mask[start:end, prev_start:start] = float("-inf")

    return mask.unsqueeze(0).unsqueeze(0)


def compress_hidden(
    hidden_states: torch.Tensor,
    compress_ratio: int,
) -> torch.Tensor:
    if hidden_states.shape[1] <= compress_ratio:
        return hidden_states.mean(dim=1, keepdim=True)
    seq_len = hidden_states.shape[1]
    n_compress = seq_len // compress_ratio
    compressed = seq_len // n_compress
    compressed_len = compressed * n_compress
    truncated = hidden_states[:, :compressed_len, :]
    compressed = truncated.view(hidden_states.shape[0], n_compress, compress_ratio, hidden_states.shape[2])
    return compressed.mean(dim=2)
