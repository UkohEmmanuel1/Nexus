# Context Extension

Extend the model's context window from 4,096 tokens to 1,048,576 (1M+) tokens using YaRN scaling, NTK-aware scaling, sliding window attention, and hierarchical memory.

## Theory

### RoPE Limitation
Standard RoPE precomputes frequencies for a fixed `max_seq_len`. Beyond this length, position embeddings become untrained and perplexity degrades rapidly. Context extension methods modify the frequency computation to handle longer sequences.

### YaRN (Yet another RoPE extensioN)

YaRN applies per-dimension interpolation with a smooth ramp between frequency bands:

```
For each dimension pair (2i, 2i+1):
  wavelength_i = 2 * pi * theta^(2i / D)
  if wavelength_i < original_max_seq_len:
    interpolate: t' = t / scaling_factor
  else:
    extrapolate: t' = t
  ramp = sigmoid(beta_fast * (1 - alpha)) * sigmoid(beta_slow * (alpha - 1))
  smooth_factor = scaling_factor - (scaling_factor - 1) * ramp
  t' = t / smooth_factor
```

Parameters:
- `scaling_factor`: target extension ratio (e.g., 256 for 4K → 1M)
- `beta_fast` (default 32): controls transition speed for short wavelengths
- `beta_slow` (default 1): controls transition speed for long wavelengths

### NTK-aware Scaling

Adjusts the base frequency theta to maintain relative position resolution:

```
theta' = theta * scaling_factor^(D / (D - 2))
```

This prevents "rope collapse" where high-frequency dimensions lose the ability to distinguish adjacent positions.

## Configuration

```yaml
model:
  long_context:
    enabled: true
    max_seq_len: 1048576
    rope_scaling:
      type: yarn
      factor: 256.0
      original_max_seq_len: 4096
      beta_fast: 32
      beta_slow: 1
    sliding_window_size: 4096
    global_attention_every_n: 128
    num_global_heads: 4
    compress_ratio: 64
    memory_tokens: 64
    chunk_size: 4096
```

## Sliding Window Attention

Each token attends to a local window of `sliding_window_size` tokens:

```
Attention(t_i) = softmax(Q_i @ K_{i-W}^{i} / sqrt(d)) @ V_{i-W}^{i}
```

Benefits:
- O(n * w) memory instead of O(n^2)
- Near full-attention quality for most tasks
- Enables processing sequences up to 1M+ on single GPU

## Hierarchical Memory

For sequences exceeding the sliding window, we use hierarchical memory:

```
Input: sequence of N tokens
  │
  ├── Chunk 1 (4K) ──→ Compress (64:1) ──→ Summary Tokens (64)
  ├── Chunk 2 (4K) ──→ Compress (64:1) ──→ Summary Tokens (64)
  │   ...
  ├── Chunk M (4K) ──→ Keep full detail
  │
  └── All summary tokens + last chunk → Full attention
```

Total context capacity: `sliding_window + memory_tokens * (num_chunks - 1)`

## Progressive Training Recipe

### Step 1: Train at 4K (base)
- Standard training at `max_seq_len = 4096`
- No context scaling

### Step 2: Extend to 32K
```yaml
rope_scaling:
  type: yarn
  factor: 8.0
  original_max_seq_len: 4096
```
- Fine-tune for 500–1000 steps on 32K sequences
- 8x interpolation is well within YaRN's comfort zone

### Step 3: Extend to 128K
```yaml
rope_scaling:
  type: yarn
  factor: 32.0
  original_max_seq_len: 4096
```
- Fine-tune for 500 steps
- Enable sliding window attention

### Step 4: Extend to 512K
```yaml
rope_scaling:
  type: yarn
  factor: 128.0
```
- Enable hierarchical memory
- Fine-tune for 500–1000 steps

### Step 5: Extend to 1M
```yaml
rope_scaling:
  type: yarn
  factor: 256.0
```
- Final fine-tuning step
- Evaluate on long-context benchmarks

## Memory Management

During inference with long contexts:

```python
from src.inference.long_context import LongContextProcessor

processor = LongContextProcessor(model, tokenizer, config)

# Process long document (1M+ tokens)
answer = processor.generate_with_long_context(
    long_text=full_document,
    query="Summarize the key findings",
    max_new_tokens=512,
)
```

The processor handles:
1. Automatic chunking into 4K segments
2. Hierarchical compression
3. KV cache management across chunks
4. Memory-efficient generation

## Benchmarking

Evaluate on long-context tasks:

```python
from src.evaluation.runner import Evaluator

evaluator = Evaluator(model, tokenizer)

# Needle in a Haystack test
results = evaluator.run_benchmark("needle_in_haystack", 
    context_lengths=[4096, 32768, 131072, 524288, 1048576])

# Long document QA
results = evaluator.run_benchmark("long_doc_qa")
```

## Known Limitations

- **Perplexity gap**: Extended models may show 0.5–2.0 higher perplexity than native models at the same length
- **Training cost**: Each extension step requires 500–2000 fine-tuning steps
- **Memory hierarchy**: Summary compression loses fine-grained positional information for early chunks
