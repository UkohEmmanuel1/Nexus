"""Bring-your-own-key model routing. Provider secrets are never persisted.

All upstream URLs are fixed to trusted providers to prevent request-driven SSRF.
"""

from __future__ import annotations

import httpx


class ProviderError(Exception):
    """Sanitized provider failure, usable without a web framework."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


# Public open-weight model cards, checked against Hugging Face on 2026-10-27.
# Hosted inference requires an HF token, available quota, and model access.
HUGGINGFACE_MODELS = {
    "qwen": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "deepseek": "deepseek-ai/DeepSeek-V3.2",
    "mimo": "XiaomiMiMo/MiMo-V2.5",
}

# Fixed provider endpoints. Model identifiers may change; users supply one supported
# by their own provider account. Nothing here asserts model access or pricing.
PROVIDERS = {
    "openai": ("https://api.openai.com/v1/chat/completions", "openai"),
    "deepseek": ("https://api.deepseek.com/chat/completions", "openai"),
    "groq": ("https://api.groq.com/openai/v1/chat/completions", "openai"),
    "together": ("https://api.together.xyz/v1/chat/completions", "openai"),
    "fireworks": ("https://api.fireworks.ai/inference/v1/chat/completions", "openai"),
    "openrouter": ("https://openrouter.ai/api/v1/chat/completions", "openai"),
    "mistral": ("https://api.mistral.ai/v1/chat/completions", "openai"),
    "huggingface": ("https://router.huggingface.co/v1/chat/completions", "openai"),
    "opencode": ("https://opencode.ai/inference/openai/v1/chat/completions", "openai"),
    "xai": ("https://api.x.ai/v1/chat/completions", "openai"),
    "anthropic": ("https://api.anthropic.com/v1/messages", "anthropic"),
    "google": (
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "openai",
    ),
}

# Examples only. Use /v1/providers to discover providers, and consult the
# provider's live catalog for currently supported model names.
EXAMPLE_MODELS = {
    "openai": ["gpt-4.1-mini"],
    "deepseek": ["deepseek-chat", "deepseek-reasoner"],
    "groq": ["llama-3.3-70b-versatile"],
    "together": ["meta-llama/Llama-3.3-70B-Instruct-Turbo"],
    "fireworks": ["accounts/fireworks/models/llama-v3p3-70b-instruct"],
    "openrouter": ["deepseek/deepseek-chat"],
    "mistral": ["mistral-small-latest"],
    "huggingface": list(HUGGINGFACE_MODELS.values()),
    "opencode": ["big-pickle"],
    "xai": ["grok-3-mini"],
    "anthropic": ["claude-sonnet-4-5"],
    "google": ["gemini-2.5-flash"],
}


def resolve_model(provider: str, model: str) -> str:
    if not isinstance(provider, str) or provider not in PROVIDERS:
        raise ProviderError(400, "Unsupported provider")
    if not isinstance(model, str) or not model or len(model) > 200:
        raise ProviderError(400, "Invalid model")
    if not all(32 < ord(c) < 127 for c in model):
        raise ProviderError(400, "Invalid model")
    if provider == "huggingface" and model.lower() in {"big-pickle", "bigpickle", "big_pickle"}:
        raise ProviderError(
            400, "Big Pickle is not a verified Hugging Face model; use its OpenCode provider"
        )
    return HUGGINGFACE_MODELS.get(model, model) if provider == "huggingface" else model


async def complete_byok(
    *,
    provider: str,
    model: str,
    messages: list[dict],
    provider_key: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> dict:
    if not isinstance(provider, str) or provider not in PROVIDERS:
        raise ProviderError(400, "Unsupported provider")
    if (
        not isinstance(provider_key, str)
        or not 1 <= len(provider_key) <= 512
        or not all(32 < ord(c) < 127 for c in provider_key)
    ):
        raise ProviderError(400, "Invalid provider API key")
    model = resolve_model(provider, model)
    if (
        not isinstance(messages, list)
        or not 1 <= len(messages) <= 100
        or any(
            not isinstance(m, dict)
            or not isinstance(m.get("role"), str)
            or m.get("role") not in {"system", "user", "assistant"}
            or not isinstance(m.get("content"), str)
            or not m["content"].strip()
            for m in messages
        )
    ):
        raise ProviderError(400, "Invalid conversation messages")
    if sum(len(m["content"]) for m in messages) > 120000:
        raise ProviderError(400, "Conversation too large; clear the history")
    if type(max_tokens) is not int or not 1 <= max_tokens <= 4096:
        raise ProviderError(400, "max_tokens must be between 1 and 4096")
    if not isinstance(temperature, (int, float)) or not 0 < temperature <= 2:
        raise ProviderError(400, "temperature must be greater than 0 and at most 2")
    url, protocol = PROVIDERS[provider]
    if protocol == "anthropic":
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = [m for m in messages if m["role"] != "system"]
        payload = {
            "model": model,
            "messages": turns,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            payload["system"] = system
        headers = {
            "x-api-key": provider_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
    else:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        headers = {"Authorization": f"Bearer {provider_key}"}

    try:
        timeout = httpx.Timeout(60.0, connect=10.0, write=10.0, pool=10.0)
        async with httpx.AsyncClient(
            timeout=timeout, follow_redirects=False, trust_env=False
        ) as client:
            response = await client.post(url, json=payload, headers=headers)
    except (httpx.TimeoutException, httpx.RequestError):
        raise ProviderError(502, "Model provider unreachable") from None
    # Never propagate raw upstream errors: they can disclose user input or credentials.
    if response.status_code in (401, 403):
        raise ProviderError(401, "Provider rejected API key or model access")
    if response.status_code == 429:
        raise ProviderError(429, "Provider rate limit or quota exceeded")
    if response.status_code in (400, 404, 422):
        raise ProviderError(
            400, "Provider rejected model or conversation; check the model ID and context size"
        )
    if 300 <= response.status_code < 400:
        raise ProviderError(502, "Provider redirect refused")
    if response.status_code >= 400:
        raise ProviderError(502, f"Provider request failed (status {response.status_code})")
    try:
        data = response.json()
        if protocol == "anthropic":
            text = "".join(
                part.get("text", "") for part in data["content"] if part.get("type") == "text"
            )
        else:
            text = data["choices"][0]["message"].get("content") or ""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("missing text")
        usage = data.get("usage") or {}
        if not isinstance(usage, dict):
            usage = {}
        return {"provider": provider, "model": model, "response": text, "usage": usage}
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        raise ProviderError(502, "Unexpected model provider response") from None
