# Inference Guide

## CLI Chat
```bash
python -m src.inference.cli --model checkpoints/model.pt --tokenizer tokenizer/tokenizer.model
```

Or via the installed entry point:
```bash
nexus --model checkpoints/model.pt --tokenizer tokenizer/tokenizer.model
```

## API Server
```bash
python -m src.inference.server
curl http://localhost:8000/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"stream":true}'
```

Or via the installed entry point:
```bash
nexus-server
nexus --serve --api-key sk-your-key
```

> **Note**: The server now requires API authentication for protected endpoints. See [Deployment](deployment.md) for key management.

## Streaming
```python
from src.inference.engine import InferenceEngine

for token in engine.generate(prompt, stream=True):
    print(token, end="", flush=True)
```

## OpenAI-compatible API

The server implements OpenAI's chat completions API format, making it a drop-in replacement for any OpenAI-compatible tool.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-key" \
  -d '{
    "model": "nexus-3b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 512,
    "stream": true
  }'
```

## Web Crawl & Search

The API server exposes a full web crawling pipeline backed by `WebCrawler`, `CrawlScheduler`, and `WebIndex`:

```bash
# Crawl a URL or search results (auto-indexed in SQLite)
curl -X POST http://localhost:8000/crawl \
  -H "Authorization: Bearer sk-your-key" \
  -H "Content-Type: application/json" \
  -d '{"search_query": "Nexus LLM", "max_depth": 1}'

# Schedule recurring crawls
curl -X POST http://localhost:8000/crawl/schedule \
  -H "Authorization: Bearer sk-your-key" \
  -d '{"seed_urls": ["https://example.com"], "interval_seconds": 3600}'

# Query the indexed corpus
curl "http://localhost:8000/crawl/data?query=llm&limit=10"

# Streaming crawl results
curl -N -X POST http://localhost:8000/crawl/stream \
  -H "Authorization: Bearer sk-your-key" \
  -d '{"url": "https://example.com"}'
```

## API Key Management

```bash
# Generate a managed key
curl -X POST http://localhost:8000/v1/keys/generate \
  -H "Authorization: Bearer sk-your-key" \
  -d '{"name": "my-app"}'

# List / revoke keys
curl http://localhost:8000/v1/keys -H "Authorization: Bearer sk-your-key"
curl -X POST http://localhost:8000/v1/keys/revoke \
  -H "Authorization: Bearer sk-your-key" \
  -d '{"key": "nexus_sk_..."}'
```
