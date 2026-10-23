import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from src.inference.providers import PROVIDERS, complete_byok


def test_provider_urls_are_fixed_https():
    for url, protocol in PROVIDERS.values():
        assert url.startswith("https://")
        assert protocol in {"openai", "anthropic"}


def test_unrecognized_provider_rejected():
    with pytest.raises(HTTPException) as error:
        asyncio.run(complete_byok(
            provider="http://internal", model="test",
            messages=[{"role": "user", "content": "hello"}],
            provider_key="key",
        ))
    assert error.value.status_code == 400


def test_invalid_key_rejected():
    with pytest.raises(HTTPException) as error:
        asyncio.run(complete_byok(
            provider="deepseek", model="deepseek-chat",
            messages=[{"role": "user", "content": "hello"}],
            provider_key="bad\nheader",
        ))
    assert error.value.status_code == 400
