import logging
from typing import Generator, List, Optional, Tuple

import torch

from src.model.config import ModelConfig
from src.model.context import HierarchicalMemory, create_long_context_mask, compress_hidden
from src.inference.engine import InferenceEngine

logger = logging.getLogger(__name__)


class LongContextProcessor:
    def __init__(self, model, tokenizer, config: ModelConfig, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.chunk_size = config.long_context.chunk_size
        self.memory = HierarchicalMemory(config)

    def process_long_input(
        self, text: str, max_chunks: int = None
    ) -> Tuple[torch.Tensor, List[str]]:
        tokens = self.tokenizer.encode(text, add_bos=True)
        chunks = []
        chunk_texts = []

        for i in range(0, len(tokens), self.chunk_size):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(torch.tensor([chunk_tokens], device=self.device))
            chunk_texts.append(chunk_text)
            if max_chunks and len(chunks) >= max_chunks:
                break

        return chunks, chunk_texts

    def generate_with_long_context(
        self,
        long_text: str,
        query: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        chunks, chunk_texts = self.process_long_input(long_text)

        self.memory.clear()
        self.model.init_kv_cache(1, self.config.effective_max_seq_len, self.device, torch.float32)

        for i, (chunk_tensor, chunk_text) in enumerate(zip(chunks[:8], chunk_texts[:8])):
            _ = self.model.forward(chunk_tensor, use_cache=True)
            self.memory.add_segment(chunk_tensor, i)

        query_tokens = self.tokenizer.encode(query, add_bos=True)
        query_tensor = torch.tensor([query_tokens], device=self.device)

        output = self.model.generate(
            query_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )
        self.model.reset_kv_cache()

        return self.tokenizer.decode(output[0].tolist())

    def summarize_long_text(
        self, text: str, max_length: int = 2048
    ) -> str:
        chunks, _ = self.process_long_input(text)
        summaries = []

        for chunk in chunks[:16]:
            chunk_text = self.tokenizer.decode(chunk[0].tolist())
            prompt = f"Summarize the following text concisely:\n\n{chunk_text}\n\nSummary:"
            input_ids = self.tokenizer.encode(prompt, add_bos=True)
            input_tensor = torch.tensor([input_ids], device=self.device)

            summary_ids = self.model.generate(
                input_tensor,
                max_new_tokens=256,
                temperature=0.3,
            )
            summary = self.tokenizer.decode(summary_ids[0].tolist())
            summaries.append(summary)

        if len(summaries) > 1:
            combined = "\n".join(summaries)
            input_ids = self.tokenizer.encode(f"Summarize all:\n\n{combined}\n\nFinal summary:", add_bos=True)
            input_tensor = torch.tensor([input_ids], device=self.device)
            final_ids = self.model.generate(input_tensor, max_new_tokens=max_length, temperature=0.3)
            return self.tokenizer.decode(final_ids[0].tolist())

        return summaries[0] if summaries else ""
