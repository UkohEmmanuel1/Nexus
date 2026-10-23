# Using the Nexus API

The hosted service is a text generation API. Ask the operator for the service URL
and your own API key. Use HTTPS for all remote calls. There is no browser chat UI.

Discover the service at `GET /` and download its OpenAPI contract at
`GET /openapi.json`. The current model is listed at `GET /models`.

## Chat

Set `NEXUS_BASE_URL` and `NEXUS_API_KEY` in your server environment, then:

```python
import os
import httpx

response = httpx.post(
    os.environ["NEXUS_BASE_URL"] + "/chat",
    headers={"Authorization": "Bearer " + os.environ["NEXUS_API_KEY"]},
    json={"message": "Explain photosynthesis simply.", "max_tokens": 128},
    timeout=120,
)
response.raise_for_status()
print(response.json()["response"])
```

Messages can use Unicode, including languages other than English. Answer quality
depends on the underlying model. Each call is independent: include any needed
conversation context in `message`. The default context is 4,096 tokens including
response and template overhead. Requests outside the context budget return 422.

`POST /chat/stream` accepts the same JSON and returns server-sent events:
`data: {"token":"..."}` followed by `data: [DONE]`. Stream errors contain an
`error` field. `POST /generate` accepts `prompt` in place of `message`.

## Personal keys (operators only)

Call `POST /v1/keys/generate` with `{"name":"customer-name"}` and the admin
bearer credential. Share the returned key privately with that client; it is
returned only once. `GET /v1/keys` lists non-secret identifiers. Revoke a client's
key using `POST /v1/keys/revoke` with `{"key":"sha256:identifier-from-list"}`.
Manage onboarding through the operator; public self-registration is not provided.

## Errors and limits

| Status | Meaning | Client action |
| --- | --- | --- |
| 401 | Missing, invalid, revoked, or insufficient credential | Ask the operator for a valid personal key |
| 408 | Request upload timed out | Retry after checking the connection |
| 413 | Body exceeds 64 KiB | Reduce request size |
| 422 | Invalid inputs or context budget exceeded | Shorten the message or response budget |
| 429 | Credential or authentication failure limit reached | Honor `Retry-After` |
| 503 | Model busy or readiness failure | Honor `Retry-After` when supplied; retry with bounded backoff |
| 500 | Server error | Report the `X-Request-ID` to the operator |

Each personal key defaults to 60 requests/minute. Generation is limited to 2,048
output tokens/request and one concurrent generation per server. Capacity is
shared by clients. Keep automatic retries bounded to avoid adding overload.
