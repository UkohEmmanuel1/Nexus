# Render API deployment

The root `render.yaml` deploys a Python API with the Qwen 2.5 1.5B GGUF model
on CPU. There is no browser frontend or Node.js build. The model revision is
pinned. Model weights and hashed API credentials live on a 5 GB disk.

## Deploy

1. Push the changes to the connected GitHub repository.
2. Create or synchronize the Render Blueprint using `render.yaml`.
3. Review the paid Standard instance and persistent disk before creating them.
4. Wait for the image build, model download, and `/health` check to pass.
5. Send authenticated requests to the service's HTTPS `/chat` endpoint.

Render generates separate `NEXUS_ADMIN_KEY` and `NEXUS_API_KEY` secrets. Use the
API secret in `Authorization: Bearer <key>` or `X-API-Key` headers. Only admins
may create, list, or revoke managed keys through `/v1/keys/*`. New keys are
returned once; subsequent listings contain non-secret identifiers.

Example request body for `POST /chat`:

```json
{"message": "Explain photosynthesis.", "max_tokens": 256, "temperature": 0.7}
```

`GET /health` is public. The root `/` returns 404 because Nexus is API-only.
A successful deployment requires both a passing health check and a real
authenticated chat request. Existing frontend Render services must be removed
or disabled separately in the Render dashboard.

## Local check

Install `requirements-serve.txt`, set separate random `NEXUS_ADMIN_KEY` and
`NEXUS_API_KEY` values, set `NEXUS_MODEL_BACKEND=gguf`, and point `MODEL_PATH`
to a local GGUF file. Run `python -m src.inference.server`.
For the repaired local model in this checkout, use
`MODEL_PATH=checkpoints/qwen2.5-1.5b-instruct-q4_k_m.verified.gguf`.
The original truncated file is preserved; the verified copy matches the
publisher's SHA-256 checksum. Model weights stay excluded from Git.
Alternatively, export both secrets and run
`docker compose -f docker/docker-compose.yml up --build -d`.

## Operational limits

- The server refuses startup without an authentication secret. Query-string
  credentials are rejected, and stored managed keys are hashed.
- API bodies are capped at 64 KiB. Prompts and response token counts are bounded.
  Concurrent generation is rejected while the model is busy.
- Reading an API request body is limited to 10 seconds. Authentication attempts
  are limited to 300 per minute per connection IP, including invalid credentials.
  The IP limiter does not trust forwarded headers; behind Render's proxy it may
  apply to shared proxy traffic. Enforce per-client limits at the edge for scale.
- CORS allows no cross-origin browser clients by default. Set explicit trusted
  origins only when needed. API error responses omit submitted input and internal
  exception details; responses use a restrictive API-only content policy.
- Rate limits are per credential and process. Use one worker and instance with
  this SQLite setup; use shared storage and a distributed limiter before scaling.
- Crawling is disabled. Enabling it requires `NEXUS_ENABLE_CRAWLER=1`, admin
  authorization, and network egress filtering against DNS rebinding.
- The container runs as UID 10001. Its persistent disk must be writable by that
  UID. Local models, databases, secrets, and Git metadata are excluded from builds.
- This backend supports text chat and streaming; it does not expose vision,
  code execution, agent execution, or separate Deep Think traces.
