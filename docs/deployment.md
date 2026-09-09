# Deployment

Deploy the model for production inference using Docker, Kubernetes, or bare-metal. Includes OpenAI-compatible API server, batch processing, and scaling configuration.

## Local Inference Server

```bash
# Start the FastAPI server (via CLI entry point)
nexus --serve \
    --model checkpoints/model.pt \
    --tokenizer tokenizer/tokenizer.model \
    --host 0.0.0.0 \
    --port 8000

# Or via module
python -m src.inference.server
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions (OpenAI format) |
| `/v1/completions` | POST | Text completions |
| `/v1/embeddings` | POST | Text embeddings |
| `/v1/code/execute` | POST | Execute Python code |
| `/v1/agent/run` | POST | Run agent task |
| `/chat` | POST | Single-turn chat |
| `/chat/stream` | POST | Streaming chat |
| `/generate` | POST | Text generation |
| `/health` | GET | Health check |
| `/models` | GET | List available models |
| `/crawl` | POST | Crawl URL(s), batch URLs, or search query |
| `/crawl/schedule` | POST | Schedule periodic crawling |
| `/crawl/status` | GET | Crawl scheduler + index stats |
| `/crawl/data` | GET | Query crawled/indexed data |
| `/crawl/stream` | POST | Streaming crawl results |
| `/v1/keys/generate` | POST | Generate an API key |
| `/v1/keys` | GET | List API keys |
| `/v1/keys/revoke` | POST | Revoke an API key |

### Authentication & API Keys

The server supports two authentication modes:

1. **Static API key** — pass `--api-key sk-your-key` (or set `NEXUS_API_KEY`). All protected endpoints require it via `Authorization: Bearer <key>`, `X-API-Key`, or `api_key` query param.
2. **Managed keys** — generated and revoked through `/v1/keys/*` endpoints, stored in a SQLite-backed `KeyManager`.

```bash
# Generate a managed key (needs NEXUS_ADMIN_KEY or the static api key)
curl -X POST http://localhost:8000/v1/keys/generate \
  -H "Authorization: Bearer sk-your-key" \
  -H "Content-Type: application/json" \
  -d '{"name": "my-app"}'
# {"key": "nexus_sk_...", "name": "my-app"}

# Use the managed key
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer nexus_sk_..." \
  -d '{"model": "nexus-3b", "messages": [{"role": "user", "content": "Hello!"}]}'

# List keys
curl http://localhost:8000/v1/keys \
  -H "Authorization: Bearer sk-your-key"

# Revoke a key
curl -X POST http://localhost:8000/v1/keys/revoke \
  -H "Authorization: Bearer sk-your-key" \
  -H "Content-Type: application/json" \
  -d '{"key": "nexus_sk_..."}'
```

Unauthenticated endpoints: `/health`, `/models`, `/v1/keys/generate`, `/v1/keys`, `/crawl/status`.

### Web Crawling

```bash
# Crawl a single URL (indexed automatically)
curl -X POST http://localhost:8000/crawl \
  -H "Authorization: Bearer sk-your-key" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'

# Crawl a batch of URLs
curl -X POST http://localhost:8000/crawl \
  -d '{"batch_urls": ["https://a.com", "https://b.com"], "max_depth": 2}'

# Crawl DuckDuckGo search results
curl -X POST http://localhost:8000/crawl \
  -d '{"search_query": "Nexus LLM"}' \
  -H "Authorization: Bearer sk-your-key"

# Schedule recurring crawls
curl -X POST http://localhost:8000/crawl/schedule \
  -d '{"seed_urls": ["https://example.com"], "interval_seconds": 3600, "max_depth": 1}'

# Query the indexed web corpus
curl "http://localhost:8000/crawl/data?query=llm&limit=10"

# Scheduler + index stats
curl http://localhost:8000/crawl/status
```

### Example Request
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nexus-3b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
    "max_tokens": 512,
    "stream": true
  }'
```

## Docker Deployment

### Build
```bash
docker build -t nexus:latest -f docker/Dockerfile .
```

### Run
```bash
docker run -d --gpus all \
  -p 8000:8000 \
  -v /path/to/models:/models \
  -e MODEL_PATH=/models/model.pt \
  -e TOKENIZER_PATH=/models/tokenizer.model \
  nexus:latest
```

### docker-compose
```bash
docker-compose -f docker/docker-compose.yml up -d
```

## Kubernetes Deployment

```bash
# Deploy
kubectl apply -f kubernetes/

# Scale
kubectl scale deployment nexus --replicas=3

# Check status
kubectl get pods -l app=nexus
```

### Files
| File | Purpose |
|------|---------|
| `kubernetes/deployment.yaml` | Main deployment with GPU resource limits |
| `kubernetes/service.yaml` | Load-balanced service (ClusterIP) |
| `kubernetes/hpa.yaml` | Horizontal Pod Autoscaler |

### Autoscaling
The HPA automatically scales replicas based on CPU/memory usage:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: nexus-hpa
spec:
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Performance Optimization

### Batch Inference
```python
from src.inference.engine import InferenceEngine

engine = InferenceEngine(model, tokenizer)

# Batch processing
results = engine.generate_batch(
    prompts=["What is 2+2?", "What is 3+3?", "What is 4+4?"],
    max_new_tokens=50,
    batch_size=8,
)
```

### Quantization
```python
# Load quantized model
model = Transformer.load_quantized(
    "checkpoints/model_8bit.pt",
    quantization="int8",
)
```

### Memory Optimization
- KV cache: 0.5–2 GB for 4K context (depends on model size)
- Model weights: 4 GB (3B fp32), 1.3 GB (3B int8)
- Recommended: 2× model size in system memory + 1.5× in GPU memory

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `checkpoints/model.pt` | Path to model weights |
| `TOKENIZER_PATH` | `tokenizer/tokenizer.model` | Path to tokenizer |
| `DEVICE` | `cuda` | Compute device |
| `DTYPE` | `bfloat16` | Model dtype |
| `MAX_SEQ_LEN` | `4096` | Max sequence length |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
| `NEXUS_API_KEY` | unset | Static API key for protected endpoints |
| `NEXUS_ADMIN_KEY` | unset | Admin key for key management endpoints |

## Monitoring

```bash
# Health check
curl http://localhost:8000/health

# Metrics (if Prometheus enabled)
curl http://localhost:8000/metrics
```

Metrics include:
- Request latency (p50, p95, p99)
- Token generation throughput (tokens/s)
- GPU utilization
- Active request count
- Model load time
