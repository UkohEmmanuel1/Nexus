# Inference Guide

## CLI Chat
```bash
python -m src.inference.cli --model checkpoints/model.pt --tokenizer tokenizer/tokenizer.model
```

## API Server
```bash
python -m src.inference.server
curl http://localhost:8000/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"stream":true}'
```

## Streaming
```python
from src.inference.engine import InferenceEngine

for token in engine.generate(prompt, stream=True):
    print(token, end="", flush=True)
```

## Batch Inference
```bash
python -m src.inference.batch --input queries.jsonl --output responses.jsonl
```

## Quantization
```bash
python -m src.inference.quantize --model checkpoints/model.pt --method int4
```

## OpenAI-compatible API

The server implements OpenAI's chat completions API format, making it a drop-in replacement for any OpenAI-compatible tool.
