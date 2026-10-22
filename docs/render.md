# Render deployment

The root `render.yaml` deploys one Docker web service serving the static Next.js
console and the Python API from the same HTTPS origin. It uses the Qwen 2.5 1.5B
GGUF model with llama.cpp on CPU, not an untrained Nexus checkpoint. The model
revision is pinned. Model weights and hashed API credentials live on a 5 GB disk.

## Deploy

1. Push the reviewed changes to the connected GitHub repository.
2. In Render, create a Blueprint from that repository and its `render.yaml`.
3. Review the paid Standard instance and persistent disk before creating them.
   This model does not fit the free instance. Increase memory if needed after
   observing actual usage; CPU inference is intended for light traffic.
4. Wait for the Docker build, initial model download, and `/health` check to pass.
5. Open the service's HTTPS URL. Retrieve `NEXUS_ADMIN_KEY` from Render's secret
   environment settings, open API Keys in the console, and create a user key.
   Give users their own API keys; keep the admin credential private.

The browser keeps credentials in memory. Reloading clears them. Generated keys
are shown once; the key list contains non-secret identifiers for revocation.
Render generates separate random admin and API secrets. Never put secrets in Git,
URLs, screenshots, or frontend build variables.

## Local check

Install `requirements-serve.txt`, then set `NEXUS_ADMIN_KEY` and `NEXUS_API_KEY`
to separate random values, `NEXUS_MODEL_BACKEND=gguf`, and `MODEL_PATH` to a local
GGUF file. Run `npm ci` and `npm run build` in `frontend`, set
`NEXUS_STATIC_DIR=frontend/out`, and run `python -m src.inference.server`.
Open `http://localhost:8000`. For Docker, export both secrets and run
`docker compose -f docker/docker-compose.yml up --build -d`.

## Security and operational limits

- Startup fails without an authentication secret. Key management requires the
  admin secret; managed keys cannot create, list, or revoke credentials.
- Credentials are accepted only in Authorization or X-API-Key headers. Legacy
  plaintext database keys are migrated to SHA-256 digests at startup. Back up
  existing databases securely and rotate keys if old backups were exposed.
- API bodies are capped at 64 KiB; prompts, response tokens, and sampling values
  are bounded. Concurrent generation is rejected while the model is busy.
- Rate limits are per credential and per process. Keep one worker/instance with
  this SQLite setup. Use a shared credential store, distributed limiter, and
  separate model serving before scaling to multiple instances.
- Crawling is disabled by default, including read access to the crawl index.
  Enabling it requires an admin and `NEXUS_ENABLE_CRAWLER=1`. URL checks reject
  private/reserved addresses and redirects, but network egress filtering is still
  required against DNS rebinding before enabling crawling on an internet server.
- The container runs as UID 10001. Its persistent disk must be writable by that
  UID. The image excludes local models, databases, environment files, and Git.
- The GGUF backend supports text chat and streaming. It does not provide vision,
  code execution, agent execution, or a separate Deep Think trace.

Deployment is successful only after Render reports the service live and real
authenticated chat succeeds. A successful local build alone is insufficient.
