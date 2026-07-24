# Deployment

Deploy the model for production inference using Docker, Kubernetes, or bare-metal. Includes OpenAI-compatible API server, batch processing, and scaling configuration.

## Local Inference Server

```bash
# Start the FastAPI server
python -m src.inference.server \
    --model checkpoints/model.pt \
    --tokenizer tokenizer/tokenizer.model \
    --host 0.0.0.0 \
    --port 8000
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions (OpenAI format) |
| `/v1/completions` | POST | Text completions |
| `/v1/embeddings` | POST | Text embeddings |
| `/v1/code/execute` | POST | Execute Python code |
| `/v1/agent/run` | POST | Run agent task |
| `/health` | GET | Health check |

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
docker build -t nexus:latest -f deploy/Dockerfile .
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
docker-compose -f deploy/docker-compose.yml up -d
```

## Kubernetes Deployment

```bash
# Deploy
kubectl apply -f deploy/kubernetes/

# Scale
kubectl scale deployment nexus --replicas=3

# Check status
kubectl get pods -l app=nexus
```

### Files
| File | Purpose |
|------|---------|
| `deploy/kubernetes/deployment.yaml` | Main deployment with GPU resource limits |
| `deploy/kubernetes/service.yaml` | Load-balanced service (ClusterIP) |
| `deploy/kubernetes/hpa.yaml` | Horizontal Pod Autoscaler |

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
