import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .context import HierarchicalMemory, create_long_context_mask
from .rmsnorm import RMSNorm
from .rope import precompute_freqs_cis
from .transformer_block import TransformerBlock


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([
            TransformerBlock(i, config) for i in range(config.n_layers)
        ])
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self._freqs_cis = None
        self._init_weights()
        self._setup_freqs_cis()

        self.memory = HierarchicalMemory(config) if config.long_context.enabled else None

    def _setup_freqs_cis(self, device: torch.device = None):
        lc = self.config.long_context
        max_len = self.config.effective_max_seq_len
        if lc.enabled and lc.rope_scaling:
            self._freqs_cis = precompute_freqs_cis(
                self.config.head_dim,
                max_len,
                self.config.rope_theta,
                device,
                scaling_type=lc.rope_scaling.type,
                scaling_factor=lc.rope_scaling.factor,
                original_max_seq_len=lc.rope_scaling.original_max_seq_len,
                beta_fast=lc.rope_scaling.beta_fast,
                beta_slow=lc.rope_scaling.beta_slow,
            )
        else:
            self._freqs_cis = precompute_freqs_cis(
                self.config.head_dim,
                max_len * 2,
                self.config.rope_theta,
                device,
            )

    def _init_weights(self):
        std = self.config.init_std
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=std)

    def init_kv_cache(self, batch_size: int, max_seq_len: int, device: torch.device, dtype: torch.dtype):
        for layer in self.layers:
            layer.attention.init_kv_cache(batch_size, max_seq_len, device, dtype)

    def reset_kv_cache(self):
        for layer in self.layers:
            layer.attention.reset_kv_cache()

    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        attention_mask: torch.Tensor = None,
        use_cache: bool = False,
        memory_tokens: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = input_ids.shape[1]
        device = input_ids.device

        if self._freqs_cis is None or self._freqs_cis.device != device:
            self._setup_freqs_cis(device)

        freqs_cis = self._freqs_cis.to(device=device)

        h = self.token_embedding(input_ids)

        if memory_tokens is not None:
            h = torch.cat([memory_tokens, h], dim=1)

        total_len = h.shape[1]
        mask = None
        if seq_len > 1 and start_pos == 0:
            if self.config.long_context.enabled and total_len > self.config.long_context.sliding_window_size:
                mask = create_long_context_mask(
                    seq_len,
                    self.config.long_context.chunk_size,
                    self.config.long_context.memory_tokens,
                    memory_tokens is not None,
                    device,
                    h.dtype,
                )
            else:
                mask = torch.full(
                    (total_len, total_len), float("-inf"), device=device, dtype=h.dtype
                )
                mask = torch.triu(mask, diagonal=1)
                if start_pos > 0:
                    mask = torch.cat(
                        [torch.zeros(total_len, start_pos, device=device, dtype=h.dtype), mask],
                        dim=-1,
                    )
                mask = mask.unsqueeze(0).unsqueeze(0)

        total_aux_loss = 0.0
        for layer in self.layers:
            h, aux_loss = layer(h, freqs_cis, start_pos, mask, use_cache)
            total_aux_loss = total_aux_loss + aux_loss

        if memory_tokens is not None:
            h = h[:, memory_tokens.shape[1]:, :]

        h = self.norm(h)
        logits = self.output(h)

        return logits, total_aux_loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        thinking_mode: bool = False,
        thinking_budget: int = 2048,
        return_thinking: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, str]:
        self.eval()
        batch_size = input_ids.shape[0]
        device = input_ids.device

        self.init_kv_cache(
            batch_size,
            self.config.effective_max_seq_len,
            device,
            next(self.parameters()).dtype,
        )

        generated = input_ids.clone()
        prompt_len = input_ids.shape[1]
        start_pos = 0
        thinking_text = ""
        in_thinking = False

        if thinking_mode:
            think_token = torch.tensor([[self.config.thinking.thinking_token_id]], device=device)
            generated = torch.cat([generated, think_token], dim=-1)
            in_thinking = True
            thinking_budget_remaining = thinking_budget

        for step in range(max_new_tokens + (thinking_budget if thinking_mode else 0)):
            if start_pos == 0:
                logits, _ = self.forward(
                    generated[:, start_pos:],
                    start_pos=0,
                    use_cache=True,
                )
            else:
                logits, _ = self.forward(
                    generated[:, -1:],
                    start_pos=start_pos,
                    use_cache=True,
                )

            next_logits = logits[:, -1, :]

            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in generated[i].tolist():
                        if token_id in (pad_token_id, eos_token_id):
                            continue
                        if next_logits[i, token_id] < 0:
                            next_logits[i, token_id] *= repetition_penalty
                        else:
                            next_logits[i, token_id] /= repetition_penalty

            next_logits = next_logits / temperature

            if top_k > 0:
                values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < values[:, -1:]] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                for i in range(batch_size):
                    indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                    next_logits[i, indices_to_remove] = float("-inf")

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)
            start_pos += 1

            if thinking_mode and in_thinking:
                thinking_budget_remaining -= 1
                if next_token.item() == self.config.thinking.end_think_token_id or thinking_budget_remaining <= 0:
                    in_thinking = False
                    think_end = torch.tensor([[self.config.thinking.end_think_token_id]], device=device)
                    generated = torch.cat([generated, think_end], dim=-1)
                    start_pos += 1
                    continue

            if not in_thinking and (next_token == eos_token_id).any():
                break

        self.reset_kv_cache()

        if return_thinking:
            return generated, thinking_text
        return generated

    def generate_with_thinking(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        thinking_budget: int = 2048,
    ) -> tuple[torch.Tensor, str]:
        return self.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            thinking_mode=True,
            thinking_budget=thinking_budget,
            return_thinking=True,
        )

    @torch.no_grad()
    def deep_think(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        n_hypotheses: int = 3,
    ) -> tuple[torch.Tensor, str]:
        candidates = []
        for _ in range(n_hypotheses):
            t = temperature * (0.8 + 0.4 * torch.rand(1).item())
            out = self.generate(
                input_ids.clone(),
                max_new_tokens=max_new_tokens,
                temperature=t,
                top_p=0.9,
                top_k=50,
            )
            candidates.append(out)

        scores = []
        for cand in candidates:
            with torch.no_grad():
                logits, _ = self.forward(cand)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = cand[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction='sum',
                )
                scores.append(-loss.item() / max(cand.shape[1] - 1, 1))

        best_idx = torch.tensor(scores).argmax().item()
        return candidates[best_idx], f"Deep Think: evaluated {n_hypotheses} hypotheses, selected #{best_idx + 1}"
