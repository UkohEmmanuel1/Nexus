# Vonk multi-provider bring-your-own-key (BYOK)

Vonk can route chat to supported model APIs without hosting the external model
or storing the user's provider API key. The existing `/chat` endpoint still
uses the locally hosted Qwen GGUF model.

## Supported providers

OpenAI, Anthropic, Google Gemini, DeepSeek, Groq, Together AI, Fireworks AI,
OpenRouter, Mistral, Hugging Face Inference Providers, and xAI.

Providers may offer open-weight models (such as Llama, Qwen, DeepSeek and
Mistral families) or proprietary hosted models. Model availability, names,
costs and licenses differ and can change. A provider API key does not
automatically grant access to every model.

## API

The Vonk API itself requires the regular Vonk bearer key, while the selected
upstream provider's key must be sent in the `X-Provider-Key` header.

```bash
curl -X POST "https://YOUR-NEXUS-HOST/v1/chat/byok" \
  -H "Authorization: Bearer YOUR_NEXUS_API_KEY" \
  -H "X-Provider-Key: YOUR_PROVIDER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"provider":"deepseek","model":"deepseek-chat","messages":[{"role":"user","content":"Explain attention in transformers."}],"max_tokens":256}'
```

List supported provider IDs and illustrative (not guaranteed current) model IDs:

```bash
curl "https://YOUR-NEXUS-HOST/v1/providers" \
  -H "Authorization: Bearer YOUR_NEXUS_API_KEY"
```

Try `provider=groq` with a Groq-hosted Llama model or
`provider=google` with a Gemini model and the corresponding API key.
For different providers, update both the provider and model fields.

## Security and operations

- BYOK secrets are only read from request headers and forwarded to a fixed,
  HTTPS provider endpoint. Vonk does not save or echo them.
- Never put provider keys in a browser app's source, localStorage, URL,
  logs or a Git commit. Keep Vonk and provider credentials separately.
- Use HTTPS and a trusted Vonk server. Its operator technically can access
  in-transit credentials; for untrusted deployments, call providers directly.
- Vonk enforces authentication and per-key rate limiting as before.
- Invalid upstream keys, quota errors and provider failures produce sanitized
  errors. The connector does not guarantee provider-level spending limits.
- No arbitrary base URLs or model download commands are accepted.
- Streaming, tool calls and multimodal provider inputs are not included in
  this initial BYOK endpoint. It currently supports text chat only.
- Each provider sets its own billing. For production, add per-user usage
  budgets, encrypted optional storage (only with opt-in), audit metadata
  without prompts or secrets, and abuse protection.

The CLI currently supports native Vonk models; use curl or any HTTP client to
access the BYOK service. A dedicated interactive BYOK CLI can be added later.
