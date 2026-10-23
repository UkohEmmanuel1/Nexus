# Render API deployment

The root `render.yaml` deploys a Python API with the Qwen 2.5 1.5B GGUF model
on CPU. There is no browser frontend or Node.js build. The model revision is
pinned. Model weights and hashed API credentials live on a 5 GB disk.

## Deploy

1. Push the changes to the connected GitHub repository.
2. Create or synchronize the Render Blueprint using `render.yaml`.
   The Blueprint tracks `main` and deploys after CI checks pass. Synchronize the
   Blueprint to apply configuration changes to an existing service.
3. Review the paid Standard instance and persistent disk before creating them.
4. Wait for the image build, model download, and `/ready` check to pass.
5. Send authenticated requests to the service's HTTPS `/chat` endpoint.

Render generates separate `NEXUS_ADMIN_KEY` and `NEXUS_API_KEY` secrets. Use the
API secret in `Authorization: Bearer <key>` or `X-API-Key` headers. Only admins
may create, list, or revoke managed keys through `/v1/keys/*`. New keys are
returned once; subsequent listings contain non-secret identifiers.

Example request body for `POST /chat`:

```json
{"message": "Explain photosynthesis.", "max_tokens": 256, "temperature": 0.7}
```

`GET /health` is public liveness. `GET /ready` also checks credential storage.
The root `/` returns API discovery JSON. `GET /openapi.json` provides the public
API contract, including bearer authentication requirements. No frontend is bundled.
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
  Concurrent generation is rejected with 503 and `Retry-After: 5` while the model
  is busy. Clients should retry with backoff and jitter, with a maximum attempt count.
- Reading an API request body is limited to 10 seconds. Authentication attempts
  failures are limited to 300 per minute per connection IP. Successful users do
  not consume this shared proxy budget. Forwarded IP headers are not trusted;
  enforce per-client abuse protection at the edge for larger traffic volumes.
- CORS allows no cross-origin browser clients by default. Set explicit trusted
  origins only when needed. API error responses omit submitted input and internal
  exception details; responses use a restrictive API-only content policy.
- Rate limits are per credential and process. Use one worker and instance with
  this SQLite setup; use shared storage and a distributed limiter before scaling.
  `NEXUS_RATE_LIMIT` defaults to 60 authenticated requests/minute/credential.
- Crawling is disabled. Enabling it requires `NEXUS_ENABLE_CRAWLER=1`, admin
  authorization, and network egress filtering against DNS rebinding.
- The container runs as UID 10001. Its persistent disk must be writable by that
  UID. Local models, databases, secrets, and Git metadata are excluded from builds.
- This backend supports text chat and streaming; it does not expose vision,
  code execution, agent execution, or separate Deep Think traces.

## Launch and operation

See [public API usage](api.md) to issue personal keys and onboard clients.
Never distribute the master or admin secret to users. Give each client a managed
key that can be revoked independently. Browser integrations should keep keys in
their own server, rather than embedding credentials in public JavaScript.

Run the deployment smoke check from a trusted shell with `NEXUS_BASE_URL` set
to the HTTPS origin and `NEXUS_API_KEY` set to a client key:

```bash
python scripts/check_production.py
```

The check verifies readiness, the API contract, rejection of unauthenticated chat,
and a real model response. It does not print credentials or prompts. A health
check alone does not verify model output. GitHub CI now runs on `main`, tests the
code, and builds/imports the actual non-root serving container.

Send application logs to your operational log service and monitor `/ready` with
an external uptime monitor. Requests include a generated `X-Request-ID` and
logs contain status and timing, without request bodies, keys, or query strings.
Streaming duration in the request log measures response setup, not full generation.

Back up `/var/data/nexus_keys.db` using SQLite's online backup API or a stopped
service snapshot; copying a live SQLite file is not a consistent backup strategy.
Keep backups encrypted and test restoration before onboarding paying users.
To roll back, deploy the previous known-good commit in the Render dashboard.
Retain the disk and secrets so existing personal keys continue to work.

This deployment can be reached internationally over HTTPS, but its single CPU
model serves one generation at a time. Measure latency and overload rates with
real prompts before promising a user count or availability target. Render disks
prevent horizontal scaling and zero-downtime deployments. Larger demand requires
an external database/limiter and multiple inference replicas, or a managed model
provider. No global capacity or uptime guarantee has been established.
