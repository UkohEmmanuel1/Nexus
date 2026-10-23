# Deployment

See [the Render deployment guide](render.md) for the supported secured chat
configuration, environment variables, operational limits, and publishing steps.
The root `render.yaml` builds the API and a CPU GGUF inference server into
one non-root Docker container. Docker Compose is also available under `docker/`.

## HTTP API

| Endpoint | Method | Authorization |
| --- | --- | --- |
| `/health` | GET | Public health check |
| `/models` | GET | Public model listing |
| `/chat` | POST | API key |
| `/chat/stream` | POST | API key; server-sent events |
| `/generate` | POST | API key |
| `/v1/keys/generate` | POST | Admin key |
| `/v1/keys` | GET | Admin key; returns identifiers, never raw keys |
| `/v1/keys/revoke` | POST | Admin key; accepts a key identifier in `key` |
| `/crawl`, `/crawl/stream`, `/crawl/schedule` | POST | Admin key; disabled by default |
| `/crawl/status`, `/crawl/data` | GET | Admin key; disabled by default |

Send credentials in `Authorization: Bearer <key>` or `X-API-Key` headers.
Query-string credentials are rejected. Configure a secret before startup.

Example authenticated chat body:

```json
{"message": "Explain photosynthesis.", "max_tokens": 256, "temperature": 0.7}
```

The current HTTP server does not implement OpenAI-compatible completion,
embedding, agent execution, or code execution endpoints. The Python research
framework has additional components that are not exposed by this deployment.
The legacy Kubernetes templates need separate cluster provisioning and secrets;
the Render Blueprint is the supported deployment path for this release.
