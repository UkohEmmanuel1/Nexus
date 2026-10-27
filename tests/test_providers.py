import asyncio

import pytest

from src.inference.providers import PROVIDERS, ProviderError, complete_byok


@pytest.mark.parametrize("body", [{}, {"choices": []}, {"choices": [{"message": {"content": ""}}]}])
def test_empty_provider_responses_are_rejected(monkeypatch, body):
    import httpx

    actual_client = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=body))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: actual_client(transport=transport, **kw))
    with pytest.raises(ProviderError) as error:
        asyncio.run(
            complete_byok(
                provider="huggingface",
                model="qwen",
                provider_key="test_key",
                messages=[{"role": "user", "content": "hello"}],
            )
        )
    assert error.value.status_code == 502


def test_provider_timeouts_are_sanitized(monkeypatch):
    import httpx

    def timeout(request):
        raise httpx.ReadTimeout("private upstream details", request=request)

    actual_client = httpx.AsyncClient
    transport = httpx.MockTransport(timeout)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: actual_client(transport=transport, **kw))
    with pytest.raises(ProviderError) as error:
        asyncio.run(
            complete_byok(
                provider="huggingface",
                model="mimo",
                provider_key="test_key",
                messages=[{"role": "user", "content": "hello"}],
            )
        )
    assert error.value.status_code == 502
    assert "private" not in error.value.detail


def test_provider_urls_are_fixed_https():
    for url, protocol in PROVIDERS.values():
        assert url.startswith("https://") or url == "http://127.0.0.1:11434/v1/chat/completions"
        assert protocol in {"openai", "anthropic"}


def test_unrecognized_provider_rejected():
    with pytest.raises(ProviderError) as error:
        asyncio.run(
            complete_byok(
                provider="http://internal",
                model="test",
                messages=[{"role": "user", "content": "hello"}],
                provider_key="key",
            )
        )
    assert error.value.status_code == 400


def test_invalid_key_rejected():
    with pytest.raises(ProviderError) as error:
        asyncio.run(
            complete_byok(
                provider="deepseek",
                model="deepseek-chat",
                messages=[{"role": "user", "content": "hello"}],
                provider_key="bad\nheader",
            )
        )
    assert error.value.status_code == 400
