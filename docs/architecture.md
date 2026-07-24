# System Architecture

## Overview

Nexus is a decoder-only Transformer language model designed for scalable pretraining, fine-tuning, and inference across model sizes from 1B to 70B+ parameters. The system is organized into six layers:

```
┌──────────────────────────────────────────────────────┐
│                    API Layer                          │
│  CLI  │  FastAPI  │  vLLM  │  Streaming  │  Batch    │
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────┐
│                  Inference Engine                     │
│  Generation Loop  │  KV Cache  │  Thinking  │  RAG   │
│  Structured Output │  Sampling  │  Quantization      │
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────┐
│                   Model Core                          │
│  Transformer  │  MoE  │  GQA Attention  │  SwiGLU   │
│  RoPE  │  RMSNorm  │  Vision Encoder  │  Context    │
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────┐
│                 Training Pipeline                     │
│  Trainer  │  FSDP/DeepSpeed  │  Checkpoint  │  Eval  │
│  SFT  │  LoRA  │  DPO  │  Reasoning  │  RLHF        │
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────┐
│                  Data Pipeline                        │
│  Discovery  │  Download  │  Clean  │  Tokenize  │ Pack│
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────┐
│               Agent System                            │
│  Planner  │  Orchestrator  │  ReAct  │  Tools        │
│  Code Executor  │  Search Grounding  │  File Search  │
└──────────────────────────────────────────────────────┘
```

## Module Map

### `src/model/` — Core Model
| File | Responsibility |
|------|---------------|
| `config.py` | `ModelConfig`, `MoEConfig`, `LongContextConfig`, `ThinkingConfig` |
| `transformer.py` | Full Transformer model, `generate()`, `generate_with_thinking()`, `deep_think()` |
| `transformer_block.py` | Single transformer block (attention + FFN with residual) |
| `attention.py` | Grouped-Query Attention with KV cache, sliding window mask |
| `feedforward.py` | SwiGLU FFN, MoE with top-k routing, aux + z loss |
| `rmsnorm.py` | Root Mean Square Normalization |
| `rope.py` | Rotary Position Embeddings with YaRN + NTK-aware scaling |
| `context.py` | Hierarchical memory, compress hidden, long context mask creation |
| `vision_encoder.py` | ViT encoder + Q-Former projector for multimodal |
| `multimodal_embedding.py` | Text + image embedding fusion |

### `src/tokenizer/` — Tokenization
| File | Responsibility |
|------|---------------|
| `tokenizer.py` | SentencePiece inference wrapper, encode/decode |
| `train.py` | Tokenizer training script (BPE/Unigram, 128k vocab) |

### `src/data/` — Data Pipeline
| File | Responsibility |
|------|---------------|
| `pipeline.py` | Dataset loading, transforms, train/val/test split |
| `tokenize_dataset.py` | Tokenization + sequence packing |

### `src/training/` — Training
| File | Responsibility |
|------|---------------|
| `trainer.py` | Training loop, AdamW, cosine LR, checkpointing |
| `distributed.py` | FSDP wrap, DeepSpeed init, multi-process setup |
| `reasoning_trainer.py` | Thinking-aware loss, PPO-based RL |

### `src/finetuning/` — Fine-tuning
| File | Responsibility |
|------|---------------|
| `sft.py` | Supervised fine-tuning |
| `lora.py` | LoRA + QLoRA via PEFT/bitsandbytes |
| `dpo.py` | Direct Preference Optimization |

### `src/inference/` — Inference
| File | Responsibility |
|------|---------------|
| `engine.py` | Generation loop, streaming, thinking mode, chat formatting |
| `cli.py` | Interactive chatbot |
| `server.py` | FastAPI server, OpenAI-compatible API |
| `structured.py` | JSON schema-constrained generation |
| `long_context.py` | 1M+ token chunking and summarization |

### `src/agents/` — Agent System
| File | Responsibility |
|------|---------------|
| `react.py` | Thought/Action/Observation loop |
| `function_calling.py` | `<function_call>` protocol |
| `planner.py` | Task decomposition into sub-steps |
| `orchestrator.py` | Multi-step agent coordinator |
| `tools.py` | Calculator, datetime, web search, Python REPL |
| `code_executor.py` | Sandboxed subprocess execution |
| `search_grounding.py` | Web search with result citation |
| `file_search.py` | Recursive file indexing + content search |
| `memory.py` | Conversation memory with summarization |

### `src/evaluation/` — Evaluation
| File | Responsibility |
|------|---------------|
| `runner.py` | Benchmark harness for MMLU, GSM8K, HumanEval, etc. |

### `src/rag/` — RAG
| File | Responsibility |
|------|---------------|
| `index.py` | Document index with cosine similarity |
| `retriever.py` | Query → top-k documents |
| `generator.py` | Context-augmented generation |

## Data Flow

### Training
```
Raw Text → Tokenizer → Token IDs → DataLoader → Model Forward → Loss → Backward → Optimizer → Checkpoint
```

### Inference
```
Prompt → Tokenizer → Input IDs → Model Forward → KV Cache → Sample → Output IDs → Tokenizer → Text
```

### Agent
```
Task → Planner (decompose) → Orchestrator → For each sub-task:
  Think → Act (tool use) → Observe → Synthesize → Final Answer
```

### Thinking Mode
```
Prompt → Tokenizer → <thinking> token → Model generates thinking tokens → </thinking> → <answer> → Generate answer → Output
```

## Distributed Training Strategy

| Model Size | Recommended Strategy | Min GPUs |
|-----------|---------------------|----------|
| 1B | DDP / FSDP | 1 |
| 3B | FSDP / DeepSpeed ZeRO-2 | 4 |
| 7B | FSDP / DeepSpeed ZeRO-3 | 8 |
| 13B | DeepSpeed ZeRO-3 | 16 |
| 32B | DeepSpeed ZeRO-3 + TP | 32 |
| 70B+ | 3D Parallelism (TP+PP+DP) | 64+ |
