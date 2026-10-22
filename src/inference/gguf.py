"""CPU GGUF serving adapter for quantized chat models."""

import os
from pathlib import Path


class GGUFEngine:
    device = "cpu"

    def __init__(self, model_path: str, context_size: int = 4096):
        from llama_cpp import Llama

        self.model = Llama(
            model_path=model_path,
            n_ctx=context_size,
            n_gpu_layers=0,
            n_threads=int(os.environ.get("NEXUS_THREADS", "2")),
            verbose=False,
        )
        self.tokenizer = self
        self.context_size = context_size

    @classmethod
    def from_environment(cls):
        model_path = os.environ.get("MODEL_PATH")
        if not model_path or not Path(model_path).is_file():
            from huggingface_hub import hf_hub_download

            model_path = hf_hub_download(
                repo_id=os.environ["NEXUS_MODEL_REPO"],
                filename=os.environ["NEXUS_MODEL_FILE"],
                revision=os.environ["NEXUS_MODEL_REVISION"],
            )
        return cls(model_path)

    def encode(self, text: str) -> list[int]:
        return self.model.tokenize(text.encode(), add_bos=False)

    def generate(
        self, prompt, max_new_tokens=512, temperature=0.7, stream=False, thinking_mode=False
    ):
        from fastapi import HTTPException

        if len(self.encode(prompt)) + max_new_tokens + 128 > self.context_size:
            raise HTTPException(422, "Prompt and response exceed the model context window")
        result = self.model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=temperature,
            stream=stream,
        )
        if stream:
            return (chunk["choices"][0].get("delta", {}).get("content", "") for chunk in result)
        return result["choices"][0]["message"]["content"]
