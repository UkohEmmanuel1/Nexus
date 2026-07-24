# Model Design

## Architecture Overview

Modern decoder-only Transformer with pre-normalization, rotary position embeddings, grouped-query attention, SwiGLU activation, and optional Mixture-of-Experts.

### Transformer Block

```
Input
  │
  ▼
RMSNorm ──► Attention (GQA + RoPE + Sliding Window)
  │
  ├── (+) ──► Residual
  │
  ▼
RMSNorm ──► SwiGLU FFN (or MoE)
  │
  ├── (+) ──► Residual
  │
  ▼
Output
```

## Components

### RMSNorm (Root Mean Square Normalization)
```
RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
```
- Applied before each sublayer (pre-norm)
- Stabilizes training by normalizing activation magnitudes
- Computationally cheaper than LayerNorm (no mean subtraction)

### RoPE (Rotary Position Embeddings)
```
RoPE(x, pos) = x * cos(theta) + rotate(x) * sin(theta)
```
- Encodes position by rotating query and key vectors
- Captures both absolute and relative position information
- Supports context extension via YaRN and NTK-aware scaling

**YaRN Scaling:**
- Scale positions: `t' = t / factor`
- Per-dimension temperature tuning: smooth interpolation between freq bands
- Enables 4K → 1M+ token context with minimal fine-tuning

### Grouped Query Attention (GQA)
- Multiple query heads share fewer key/value heads
- Reduces KV cache size by `n_heads / n_kv_heads` during inference
- Configuration: 32 query heads, 8 KV heads (4:1 ratio)

### Sliding Window Attention
- Each token attends to only the last `window_size` tokens
- Combined with global attention every N layers for long-range dependencies
- Default window: 4096 tokens

### SwiGLU Activation
```
SwiGLU(x) = silu(W_gate * x) * (W_up * x)
FFN(x) = W_down * SwiGLU(x)
```
- Gated variant of SwiLU (SiLU)
- Consistently outperforms ReLU and GELU in transformers
- Requires 3 weight matrices vs 2 for standard FFN

## Configurations

| Model | Params | dim | n_layers | n_heads | n_kv_heads | ffn_mult | MoE |
|-------|--------|-----|----------|---------|------------|----------|-----|
| 1B | ~1.0B | 2048 | 16 | 16 | 8 | 8/3 | — |
| 3B | ~3.0B | 3200 | 26 | 32 | 8 | 8/3 | 8×224M |
| 7B | ~6.7B | 4096 | 32 | 32 | 8 | 8/3 | 8×521M |
| 13B | ~13.0B | 5120 | 40 | 40 | 8 | 8/3 | 8×1.0B |
| 32B | ~32.5B | 6656 | 60 | 52 | 8 | 8/3 | 16×1.3B |
| 70B | ~70.6B | 8192 | 80 | 64 | 8 | 8/3 | 16×2.6B |

## MoE Architecture

### Gating
```
gates = softmax(W_gate * x)
top_k_values, top_k_indices = topk(gates, k=2)
gates = top_k_values / sum(top_k_values)
```

### Expert Routing
- Each token is routed to top-2 experts
- Experts are SwiGLU FFN modules
- Output is weighted sum of expert outputs by gate scores

### Load Balancing
- **Auxiliary loss**: penalizes imbalance in token-to-expert assignment
  `aux_loss = num_experts * sum(frac * gate_weights)`
- **z-loss**: penalizes extreme logits for routing stability
  `z_loss = mean(logsumexp(gate_logits)^2)`

## Long Context Extension

### YaRN (Yet another RoPE extensioN)
```
freqs = 1 / theta^(2i/d)
t' = t / scaling_factor * ramp(dim) + t * (1 - ramp(dim))
```
- **ramp(dim)**: smooth transition between interpolated and extrapolated frequency bands
- Low frequencies (long wavelengths) are interpolated
- High frequencies (short wavelengths) remain unchanged

### NTK-aware Scaling
```
theta' = theta * scaling_factor^(D/(D-2))
```
- Prevents the "rope collapse" where high frequencies lose expressiveness
- All frequencies are scaled proportionally

### Hierarchical Memory
```
Chunk (4K) → Compress (64:1) → Summary Tokens (64 per chunk)
```
- Each 4K chunk is compressed to 64 summary tokens
- Global attention across summary tokens enables 1M+ context
- Last N chunks are kept in full detail

## Thinking Mode

### Token Format
```
[<thinking>] ...thinking tokens... [</thinking>] [<answer>] ...answer... [</answer>]
```

### Configuration
- `thinking_token_budget`: max thinking tokens (default 2048)
- `deep_think_hypotheses`: number of parallel hypotheses (default 3)
- Budget is consumed by thinking tokens; generation stops when budget exhausted or `</thinking>` emitted

### Deep Think
1. Generate N hypotheses at different temperatures
2. Score each by negative perplexity
3. Return highest-scoring hypothesis

## Weight Initialization
- All Linear layers: Normal(0, init_std) where init_std = 0.02
- Embedding: Normal(0, 0.02)
- No bias in Linear layers (following LLaMA convention)

## Dtype Support
- Training: bfloat16 (preferred) or float16
- Inference: bfloat16, float16, float32, int8 (quantized)
- Mixed precision via `torch.amp.autocast`
