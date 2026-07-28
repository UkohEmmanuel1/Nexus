from collections.abc import Generator

import torch

from src.model import Transformer


class InferenceEngine:
    def __init__(self, model: Transformer, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        stop_strings: list[str] | None = None,
        stream: bool = False,
        thinking_mode: bool = False,
        thinking_budget: int = 2048,
        return_thinking: bool = False,
    ) -> str | tuple[str, str]:
        input_ids = self.tokenizer.encode(prompt, add_bos=True)
        input_tensor = torch.tensor([input_ids], device=self.device)

        if stream:
            return self._generate_stream(
                input_tensor, max_new_tokens, temperature, top_p, top_k,
                repetition_penalty, stop_strings, thinking_mode, thinking_budget,
            )

        if thinking_mode:
            output_ids, thinking_text = self.model.generate_with_thinking(
                input_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                thinking_budget=thinking_budget,
            )
            output_text = self.tokenizer.decode(output_ids[0].tolist())

            if return_thinking:
                return output_text, thinking_text

            think_start = output_text.find("<thinking>")
            think_end = output_text.find("</thinking>")
            if think_start >= 0 and think_end >= 0:
                thinking_text = output_text[think_start + len("<thinking>"):think_end]
                clean_text = output_text[:think_start] + output_text[think_end + len("</thinking>"):]
                return clean_text

        output_ids = self.model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )

        output_text = self.tokenizer.decode(output_ids[0].tolist())
        if stop_strings:
            for stop in stop_strings:
                if stop in output_text:
                    output_text = output_text[: output_text.index(stop)]
        return output_text

    def _generate_stream(
        self,
        input_tensor: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        stop_strings: list[str] | None,
        thinking_mode: bool = False,
        thinking_budget: int = 2048,
    ) -> Generator[str, None, None]:
        self.model.init_kv_cache(
            input_tensor.shape[0],
            self.model.config.effective_max_seq_len,
            self.device,
            next(self.model.parameters()).dtype,
        )

        generated = input_tensor.clone()
        start_pos = 0
        output_text = ""
        in_thinking = thinking_mode
        thinking_remaining = thinking_budget if thinking_mode else 0

        total_budget = max_new_tokens + (thinking_budget if thinking_mode else 0)

        for _ in range(total_budget):
            if start_pos == 0:
                logits, _ = self.model.forward(generated[:, start_pos:], start_pos=0, use_cache=True)
            else:
                logits, _ = self.model.forward(generated[:, -1:], start_pos=start_pos, use_cache=True)

            next_logits = logits[:, -1, :] / temperature

            if top_k > 0:
                values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < values[:, -1:]] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                for i in range(input_tensor.shape[0]):
                    indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                    next_logits[i, indices_to_remove] = float("-inf")

            probs = torch.nn.functional.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)
            start_pos += 1

            token_str = self.tokenizer.decode([next_token.item()])

            if in_thinking:
                thinking_remaining -= 1
                if next_token.item() == self.model.config.thinking.end_think_token_id or thinking_remaining <= 0:
                    in_thinking = False
                    yield "\n"
                continue

            output_text += token_str
            yield token_str

            if next_token.item() == self.tokenizer.eos_id:
                break

            if stop_strings:
                for stop in stop_strings:
                    if stop in output_text:
                        return

        self.model.reset_kv_cache()

    def deep_think(self, prompt: str, max_new_tokens: int = 512, n_hypotheses: int = 3) -> tuple[str, str]:
        input_ids = self.tokenizer.encode(prompt, add_bos=True)
        input_tensor = torch.tensor([input_ids], device=self.device)

        output_ids, report = self.model.deep_think(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            n_hypotheses=n_hypotheses,
        )
        output_text = self.tokenizer.decode(output_ids[0].tolist())
        return output_text, report

    def chat(self, messages: list[dict], **kwargs) -> str:
        prompt = self._format_chat(messages)
        return self.generate(prompt, **kwargs)

    def _format_chat(self, messages: list[dict]) -> str:
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        formatted += "<|im_start|>assistant\n"
        return formatted
