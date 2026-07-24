# Evaluation

Benchmark the model across standard NLP tasks and long-context evaluations. The evaluation runner supports MMLU, GSM8K, HumanEval, and custom benchmarks via lm-evaluation-harness.

## Evaluation Runner (`src/evaluation/runner.py`)

```python
from src.evaluation.runner import Evaluator

evaluator = Evaluator(model, tokenizer)

# Perplexity
ppl = evaluator.evaluate_perplexity(dataset)
print(f"Perplexity: {ppl:.2f}")

# Run specific benchmark
results = evaluator.run_benchmark("mmlu", shots=5)
print(f"MMLU 5-shot: {results['accuracy']:.2%}")

# Run all
all_results = evaluator.run_all_benchmarks()
```

## Supported Benchmarks

| Benchmark | Task | Shots | Metric |
|-----------|------|-------|--------|
| Perplexity | Language modeling | 0 | Perplexity |
| MMLU | Multi-task knowledge | 5 | Accuracy |
| GSM8K | Math word problems | 8 | Exact match |
| HumanEval | Code generation | 0 | Pass@1 |
| ARC-Easy | Science Q&A | 25 | Accuracy |
| ARC-Challenge | Science Q&A | 25 | Accuracy |
| HellaSwag | Commonsense NLI | 10 | Accuracy |
| TruthfulQA | Truthfulness | 0 | MC1/MC2 |
| BBH | Big-Bench Hard | 3 | Accuracy |
| Needle-in-Haystack | Long context | 0 | Retrieval accuracy |

## Usage

### CLI
```bash
# Run all benchmarks
python -m src.evaluation.runner \
    --model checkpoints/model.pt \
    --tokenizer tokenizer/tokenizer.model \
    --benchmarks all

# Run specific benchmarks
python -m src.evaluation.runner \
    --model checkpoints/model.pt \
    --tokenizer tokenizer/tokenizer.model \
    --benchmarks mmlu,gsm8k \
    --shots 5

# Perplexity only
python -m src.evaluation.runner \
    --model checkpoints/model.pt \
    --tokenizer tokenizer/tokenizer.model \
    --benchmarks perplexity \
    --dataset data/eval/wikitext2
```

### Python API
```python
# MMLU
mmlu_results = evaluator.run_benchmark("mmlu", shots=5)
print(f"MMLU Accuracy: {mmlu_results['accuracy']:.4f}")
print(f"Per-subject:")
for subject, acc in mmlu_results['subjects'].items():
    print(f"  {subject}: {acc:.4f}")

# GSM8K
gsm8k_results = evaluator.run_benchmark("gsm8k", shots=8)
print(f"GSM8K: {gsm8k_results['accuracy']:.4f}")

# HumanEval
humaneval_results = evaluator.run_benchmark("humaneval")
print(f"HumanEval Pass@1: {humaneval_results['pass@1']:.4f}")
```

## Needle-in-a-Haystack

Evaluates long-context retrieval by inserting a specific fact into a long context and testing if the model can find it:

```python
results = evaluator.run_benchmark("needle_in_haystack",
    context_lengths=[4096, 8192, 16384, 32768, 65536, 131072],
    depth_percentages=[0, 25, 50, 75, 100],
)
```

Output:
```
Needle-in-Haystack Results:
  Context 4K:   98.2%
  Context 8K:   96.4%
  Context 16K:  92.8%
  Context 32K:  85.1%
  Context 64K:  72.3%
  Context 128K: 58.7%
```

## Custom Evaluation

```python
# Custom task
custom_results = evaluator.evaluate_custom(
    prompts=["What is 2+2?", "What is 3+3?"],
    references=["4", "6"],
    metric="exact_match",
)
```

## Configuration

```yaml
evaluation:
  benchmarks:
    - mmlu
    - gsm8k
    - humaneval
  shots:
    mmlu: 5
    gsm8k: 8
    humaneval: 0
  batch_size: 4
  output_dir: results/
```

## Results Logging

Results are saved to `results/eval_{timestamp}.json`:
```json
{
  "model": "nexus-3b",
  "timestamp": "2024-01-15T10:30:00",
  "results": {
    "mmlu": {"accuracy": 0.523, "shots": 5},
    "gsm8k": {"accuracy": 0.415, "shots": 8},
    "humaneval": {"pass@1": 0.287, "shots": 0}
  }
}
```

## Tips

1. **Use few-shot**: Benchmarks like MMLU and GSM8K require few-shot examples from the dataset.
2. **Batch inference**: Use `batch_size > 1` for faster evaluation on GPU.
3. **Perplexity baseline**: Compare against GPT-2, LLaMA, or Mistral at similar parameter counts.
4. **Focus on your domain**: Custom evaluation on domain-specific datasets matters more than general benchmarks.
