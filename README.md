# Nexus

**Open-source, scalable, production-ready Large Language Model** — designed to match frontier model capabilities including thinking-native reasoning, 1M+ token context, multimodality, code execution, and advanced agent orchestration.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](#)

---

## Key Features

| Capability | Status |
|-----------|--------|
| Thinking-native reasoning | ✅ Built-in thinking trace, controllable budget, Deep Think |
| 1M+ token context | ✅ YaRN + NTK-aware scaling, sliding window, hierarchical memory |
| Multimodal (vision) | ✅ ViT encoder + Q-Former projector |
| Code execution sandbox | ✅ Secure subprocess, AST validation, import whitelist |
| Advanced agents | ✅ Planner, orchestrator, ReAct, function calling, search grounding |
| Structured output | ✅ JSON mode, schema-constrained generation |
| MoE (Mixture of Experts) | ✅ Configurable, top-k routing, aux + z loss |
| Distributed training | ✅ FSDP + DeepSpeed ZeRO-2/3 |
| Fine-tuning | ✅ SFT, LoRA, QLoRA, DPO |

---

## Quick Start

```bash
# Install
pip install -e ".[all]"

# Train tokenizer (requires text corpus)
nexus-train --input data/corpus.txt --vocab-size 128000

# Chat with a trained model
nexus --model checkpoints/model.pt --tokenizer tokenizer/tokenizer.model

# Thinking mode
nexus --model checkpoints/model.pt --tokenizer tokenizer/tokenizer.model --thinking

# API server
nexus --serve --api-key sk-your-key

# Or install globally via npm
cd packages/nexus && npm link
nexus --model checkpoints/model.pt
```

---

## Model Sizes

| Model | Params | dim | layers | heads | kv_heads | MoE | Context |
|-------|--------|-----|--------|-------|----------|-----|---------|
| 1B | ~1B | 2048 | 16 | 16 | 8 | Optional | 4K–1M |
| 3B | ~3B | 3200 | 26 | 32 | 8 | ✓ 8 experts | 4K–1M |
| 7B | ~7B | 4096 | 32 | 32 | 8 | ✓ 8 experts | 4K–1M |
| 13B | ~13B | 5120 | 40 | 40 | 8 | ✓ 8 experts | 4K–1M |
| 32B | ~32B | 6656 | 60 | 52 | 8 | ✓ 16 experts | 4K–1M |
| 70B | ~70B | 8192 | 80 | 64 | 8 | ✓ 16 experts | 4K–1M |

---

## Architecture

```
┌──────────────────────────────────────────────┐
│              API Layer                        │
│  CLI  │  FastAPI  │  Streaming  │  Batch      │
└──────────────────┬───────────────────────────┘
                   │
┌──────────────────▼───────────────────────────┐
│           Inference Engine                    │
│  Generation  │  KV Cache  │  Thinking  │ RAG  │
└──────────────────┬───────────────────────────┘
                   │
┌──────────────────▼───────────────────────────┐
│           Model Core                          │
│  Transformer  │  MoE  │  Attention  │  FFN   │
│  RoPE  │  RMSNorm  │  SwiGLU  │  GQA         │
└──────────────────┬───────────────────────────┘
                   │
┌──────────────────▼───────────────────────────┐
│          Training Pipeline                    │
│  Trainer  │  FSDP/DeepSpeed  │  Checkpoint    │
│  SFT  │  LoRA  │  DPO  │  Reasoning          │
└──────────────────┬───────────────────────────┘
                   │
┌──────────────────▼───────────────────────────┐
│           Data Pipeline                       │
│  Discover  │  Download  │  Clean  │  Tokenize │
└──────────────────────────────────────────────┘
```

---

## Documentation Map

| Document | Audience | What It Covers |
|----------|----------|---------------|
| [Architecture](docs/architecture.md) | Engineers | System design, module map, data flows |
| [Model Design](docs/model.md) | Researchers | Architecture details, configs, components |
| [Context Extension](docs/context_extension.md) | Researchers | YaRN theory, 1M token recipe, sliding window |
| [Thinking Mode](docs/thinking_mode.md) | Developers | Thinking tokens, budget, Deep Think algorithm |
| [Multimodal](docs/multimodal.md) | Practitioners | Vision encoder, image processing, inference |
| [Code Execution](docs/code_execution.md) | Developers | Sandbox, security, allowed imports, API |
| [Agents](docs/agents.md) | AI Engineers | Planner, orchestrator, tools, search grounding |
| [Structured Output](docs/structured_output.md) | API Users | JSON mode, schema constraint, retry logic |
| [Training Guide](docs/training.md) | ML Engineers | Pretraining, FSDP, DeepSpeed, multi-node |
| [Fine-tuning Guide](docs/finetuning.md) | Practitioners | SFT, LoRA, QLoRA, DPO, thinking-aware SFT |
| [Inference Guide](docs/inference.md) | Developers | CLI, API, streaming, quantization, export |
| [API Reference](docs/api_reference.md) | All Developers | CLI flags, REST endpoints, Python API |
| [Deployment](docs/deployment.md) | DevOps | Docker, K8s, scaling, monitoring |
| [Scaling Strategy](docs/scaling.md) | Architects | 1B → 70B+ roadmap, parallelism, hardware |
| [Research Notes](docs/research_notes.md) | Researchers | Design decisions, ablations, future work |
| [Contributing](docs/contributing.md) | Contributors | Setup, PR workflow, code standards |

---

## License

MIT
