# Thinking Mode

Built-in chain-of-thought reasoning that generates intermediate thinking tokens before producing the final answer. The model natively supports thinking traces, controllable budgets, and parallel hypothesis evaluation (Deep Think).

## Overview

Thinking mode adds special tokens that trigger the model to reason step-by-step before answering:

```
Input: "What is 15 * 37?"

Output:
<thinking>
Let me break this down:
1. 15 * 30 = 450
2. 15 * 7 = 105
3. 450 + 105 = 555
</thinking>
<answer>
15 * 37 = 555
</answer>
```

## Special Tokens

| Token | ID | Purpose |
|-------|----|---------|
| `<thinking>` | 128000 | Start thinking trace |
| `</thinking>` | 128001 | End thinking trace |
| `<answer>` | 128002 | Start final answer |
| `</answer>` | 128003 | End final answer |

These IDs are configurable in `ThinkingConfig`.

## Usage

### CLI
```bash
# Basic thinking mode
python -m src.inference.cli --model model.pt --tokenizer tokenizer.model --thinking

# Control thinking budget
python -m src.inference.cli --model model.pt --tokenizer tokenizer.model \
    --thinking --thinking-budget 4096

# Deep Think mode (evaluate multiple hypotheses)
python -m src.inference.cli --model model.pt --tokenizer tokenizer.model \
    --thinking --deep-think --hypotheses 5
```

### Python API
```python
from src.inference.engine import InferenceEngine

# Basic thinking
response = engine.generate(
    prompt="What is 15 * 37?",
    thinking_mode=True,
    thinking_budget=2048,
)

# Streaming with thinking trace
for token in engine.generate(
    prompt="Solve step by step: 15 * 37",
    stream=True,
    thinking_mode=True,
):
    print(token, end="", flush=True)

# Get thinking trace separately
response, thinking_trace = engine.generate(
    prompt="Solve: 15 * 37",
    thinking_mode=True,
    return_thinking=True,
)
print(f"Thinking: {thinking_trace}")
print(f"Answer: {response}")
```

### REST API
```json
POST /v1/chat/completions
{
  "messages": [{"role": "user", "content": "What is 15 * 37?"}],
  "thinking": true,
  "thinking_budget": 2048,
  "stream": true
}
```

Response (streaming):
```
data: {"type": "thinking", "content": "Let me break this down..."}
data: {"type": "answer", "content": "15 * 37 = 555"}
```

## Deep Think Mode

Deep Think generates multiple candidate answers in parallel, scores them, and returns the best one.

### Algorithm
```
1. Generate N hypotheses at different temperatures:
   H_i = model.generate(prompt, temperature=T_i)
   
2. Score each hypothesis by negative perplexity:
   score_i = -sum(log P(token_j | H_i)) / len(H_i)
   
3. Return highest-scoring hypothesis:
   best = argmax(score_i)
```

### Usage
```python
from src.model import Transformer

# Deep think with 5 hypotheses
output, report = model.deep_think(
    input_ids,
    max_new_tokens=512,
    temperature=0.7,
    n_hypotheses=5,
)
print(report)  # "Deep Think: evaluated 5 hypotheses, selected #3"
```

### Configuration
```yaml
model:
  thinking:
    enabled: true
    thinking_token_budget: 2048
    deep_think_enabled: true
    deep_think_hypotheses: 3
```

## Thinking-Aware Fine-Tuning

For best results, fine-tune on datasets with thinking traces:

### Data Format
```json
{
  "messages": [
    {"role": "user", "content": "What is 15 * 37?"},
    {"role": "assistant", "content": "<thinking>Let me break this down...\n1. 15 * 30 = 450\n2. 15 * 7 = 105\n3. 450 + 105 = 555\n</thinking>\n<answer>15 * 37 = 555</answer>"}
  ]
}
```

### Training
```python
from src.training.reasoning_trainer import ReasoningTrainer

trainer = ReasoningTrainer(
    model,
    config={
        "thinking_loss_weight": 0.3,  # Weight thinking tokens more
        "lr": 2e-5,
    }
)
trainer.train(train_loader)
```

The `thinking_loss_weight` parameter increases the loss contribution from thinking tokens, encouraging the model to produce higher-quality reasoning traces.

## Best Practices

1. **Set appropriate budget**: Simple tasks need fewer thinking tokens (256–512). Complex math/code tasks benefit from larger budgets (2048–8192).

2. **Temperature matters**: Lower temperature (0.3–0.5) for focused reasoning; higher (0.7–1.0) for creative problem-solving.

3. **Stream thinking**: Show thinking tokens in dimmed/bracketed text during streaming to improve user experience.

4. **Deep Think for critical tasks**: Use for math, code, and factual questions where accuracy is paramount. The 3x compute cost is worth it for correctness.

5. **Combine with tools**: Thinking mode works with code execution, search grounding, and function calling.
