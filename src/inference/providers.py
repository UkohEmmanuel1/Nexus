"""Bring-your-own-key model routing. Provider secrets are never persisted.

All upstream URLs are fixed to trusted providers to prevent request-driven SSRF.
"""
from __future__ import annotations

import httpx
from fastapi import HTTPException

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
    "xai": ("https://api.x.ai/v1/chat/completions", "openai"),
    "anthropic": ("https://api.anthropic.com/v1/messages", "anthropic"),
    "google": ("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", "openai"),
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
    "huggingface": ["Qwen/Qwen2.5-72B-Instruct"],
    "xai": ["grok-3-mini"],
    "anthropic": ["claude-sonnet-4-5"],
    "google": ["gemini-2.5-flash"],
}


async def complete_byok(
    *, provider: str, model: str, messages: list[dict], provider_key: str,
    max_tokens: int = 512, temperature: float = 0.7,
) -> dict:
    if provider not in PROVIDERS:
        raise HTTPException(400, "Unsupported provider")
    if not provider_key or len(provider_key) > 512 or "\n" in provider_key or "\r" in provider_key:
        raise HTTPException(400, "Invalid provider API key")
    if not model or len(model) > 200 or not all(32 < ord(c) < 127 for c in model):
        raise HTTPException(400, "Invalid model")
    url, protocol = PROVIDERS[provider]
    if protocol == "anthropic":
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = [m for m in messages if m["role"] != "system"]
        payload = {
            "model": model, "messages": turns,
            "max_tokens": max_tokens, "temperature": temperature,
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
            "model": model, "messages": messages,
            "max_tokens": max_tokens, "temperature": temperature,
            "stream": False,
        }
        headers = {"Authorization": f"Bearer {provider_key}"}

    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=False, trust_env=False) as client:
            response = await client.post(url, json=payload, headers=headers)
    except (httpx.TimeoutException, httpx.RequestError):
        raise HTTPException(502, "Model provider unreachable") from None
    # Never propagate raw upstream errors: they can disclose user input or credentials.
    if response.status_code in (401, 403):
        raise HTTPException(401, "Provider rejected API key or model access")
    if response.status_code == 429:
        raise HTTPException(429, "Provider rate limit or quota exceeded")
    if response.status_code >= 400:
        raise HTTPException(502, f"Provider request failed (status {response.status_code})")
    try:
        data = response.json()
        if protocol == "anthropic":
            text = "".join(part.get("text", "") for part in data["content"] if part.get("type") == "text")
        else:
            text = data["choices"][0]["message"].get("content") or ""
        usage = data.get("usage", {})
        return {"provider": provider, "model": model, "response": text, "usage": usage}
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        raise HTTPException(502, "Unexpected model provider response") from None
