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

## Getting Started & Usage

### 1. Prerequisites & Installation

Ensure you have **Python 3.10+** and **Git** installed on your machine or laptop.

```bash
# Clone repository
git clone https://github.com/UkohEmmanuel1/Nexus.git
cd Nexus

# Create and activate virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On macOS / Linux:
source .venv/bin/activate

# Install dependencies
pip install -e ".[all]"
# (Or for lightweight inference only: pip install -e ".[inference]")
```

---

### 2. Running on Your Laptop / Local Machine

Nexus can run on both CPU and CUDA-enabled GPUs. When running on a standard laptop without an NVIDIA GPU, add `--device cpu`.

#### Option A: Interactive CLI Chat
```bash
# Standard chat (CPU)
nexus --model checkpoints/model.pt --tokenizer tokenizer/tokenizer.model --device cpu

# Thinking Mode (with reasoning trace enabled)
nexus --model checkpoints/model.pt --tokenizer tokenizer/tokenizer.model --device cpu --thinking
```
*In chat mode: type your message and press Enter. Use `/clear` to reset context or `exit` to quit.*

---

#### Option B: OpenAI-Compatible API Server
Start the local FastAPI server:
```bash
nexus --serve --device cpu --port 8000 --api-key sk-your-key
```

Query the completions endpoint using `curl`:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-key" \
  -d '{
    "model": "nexus",
    "messages": [{"role": "user", "content": "Explain quantum computing in simple terms."}],
    "stream": true
  }'
```

---

#### Option C: Web Frontend UI (Browser)
Nexus includes a Next.js chat interface:
1. Ensure the API server is running (`nexus --serve --device cpu`).
2. Launch the frontend:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
3. Open `http://localhost:3000` in your web browser.

---

#### Option D: Python SDK (Direct Code Integration)
```python
from src.inference.engine import InferenceEngine
from src.model import ModelConfig, Transformer
from src.tokenizer import Tokenizer
import torch

# Load configuration and model
config = ModelConfig()
model = Transformer(config)
state = torch.load("checkpoints/model.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state.get("model_state_dict", state))

tokenizer = Tokenizer("tokenizer/tokenizer.model")
engine = InferenceEngine(model, tokenizer, device="cpu")

# Stream generation
for token in engine.generate("Write a quicksort function in Python:", stream=True):
    print(token, end="", flush=True)
```

---

### 3. Training & Fine-Tuning

```bash
# Train tokenizer
nexus-train --input data/corpus.txt --vocab-size 128000

# Run evaluation benchmarks
nexus-eval --model checkpoints/model.pt --benchmarks gsm8k,humaneval
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
| [Agents](docs/agents.md) | AI Engineers | Planner, orchestrator, tools, search grounding, web crawler |
| [Structured Output](docs/structured_output.md) | API Users | JSON mode, schema constraint, retry logic |
| [Training Guide](docs/training.md) | ML Engineers | Pretraining, FSDP, DeepSpeed, multi-node |
| [Fine-tuning Guide](docs/finetuning.md) | Practitioners | SFT, LoRA, QLoRA, DPO, thinking-aware SFT |
| [Inference Guide](docs/inference.md) | Developers | CLI, API, streaming, quantization, web crawl endpoints |
| [Deployment](docs/deployment.md) | DevOps | Docker, K8s, scaling, monitoring, API keys |
| [Data Pipeline](docs/data_pipeline.md) | Engineers | Dataset discovery, download, tokenization |
| [Evaluation](docs/evaluation.md) | Researchers | Benchmarks, metrics, long-context evaluation |
| [Contributing](docs/contributing.md) | Contributors | Setup, PR workflow, code standards |

---

## License

MIT
