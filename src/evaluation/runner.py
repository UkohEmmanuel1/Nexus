import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.utils.logging import get_logger

logger = get_logger(__name__)


class Evaluator:
    def __init__(self, model: torch.nn.Module, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @torch.no_grad()
    def compute_perplexity(self, dataloader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch.get("labels", input_ids).to(self.device)

            logits, _ = self.model(input_ids)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=0,
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += (shift_labels != 0).sum().item()

        ppl = math.exp(total_loss / max(total_tokens, 1))
        return ppl

    @torch.no_grad()
    def generate_text(
        self,
        prompts: list[str],
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        **kwargs,
    ) -> list[str]:
        self.model.eval()
        results = []

        for prompt in prompts:
            input_ids = torch.tensor(
                [self.tokenizer.encode(prompt, add_bos=True)],
                device=self.device,
            )
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs,
            )
            text = self.tokenizer.decode(output_ids[0].tolist())
            results.append(text)

        return results

    def run_benchmark(self, benchmark: str, **kwargs) -> dict:
        logger.info(f"Running benchmark: {benchmark}")
        if benchmark == "mmlu":
            return self._evaluate_mmlu(**kwargs)
        elif benchmark == "gsm8k":
            return self._evaluate_gsm8k(**kwargs)
        elif benchmark == "humaneval":
            return self._evaluate_humaneval(**kwargs)
        else:
            raise ValueError(f"Unknown benchmark: {benchmark}")

    def _evaluate_mmlu(self, **kwargs) -> dict:
        try:
            from lm_eval import evaluator
            results = evaluator.simple_evaluate(
                model=self,
                tasks=["mmlu"],
                **kwargs,
            )
            return results
        except ImportError:
            logger.error("lm_eval not installed. Install with: pip install lm-eval")
            return {"error": "lm_eval not installed"}

    def _evaluate_gsm8k(self, **kwargs) -> dict:
        try:
            from lm_eval import evaluator
            results = evaluator.simple_evaluate(
                model=self,
                tasks=["gsm8k"],
                **kwargs,
            )
            return results
        except ImportError:
            logger.error("lm_eval not installed")
            return {"error": "lm_eval not installed"}

    def _evaluate_humaneval(self, **kwargs) -> dict:
        try:
            from evalplus.evaluate import evaluate
            results = evaluate(
                model=self,
                dataset="humaneval",
                **kwargs,
            )
            return results
        except ImportError:
            logger.error("evalplus not installed")
            return {"error": "evalplus not installed"}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--benchmark", type=str, default="mmlu")
    parser.add_argument("--output", type=str, default="results.json")
    args = parser.parse_args()

    from src.model import ModelConfig, Transformer
    from src.tokenizer import Tokenizer

    tokenizer = Tokenizer(args.tokenizer)
    config = ModelConfig()
    model = Transformer(config)

    state = torch.load(args.model, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])

    evaluator = Evaluator(model, tokenizer)
    results = evaluator.run_benchmark(args.benchmark)

    output_path = Path(args.output)
    output_path.write_text(json.dumps(results, indent=2))
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
