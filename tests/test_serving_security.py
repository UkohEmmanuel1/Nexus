import socket
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from src.agents.web_crawler import WebCrawler
from src.inference.gguf import GGUFEngine


@pytest.mark.parametrize("address", ["127.0.0.1", "10.0.0.1", "169.254.169.254", "::1", "0.0.0.0"])
def test_crawler_rejects_internal_dns(monkeypatch, address):
    monkeypatch.setattr(
        socket, "getaddrinfo", lambda *args, **kwargs: [(0, 0, 0, "", (address, 443))]
    )
    with pytest.raises(ValueError):
        WebCrawler._validate_public_url("https://example.com")


@pytest.mark.parametrize(
    "url", ["file:///etc/passwd", "http://user:pass@example.com", "http://example.com:22"]
)
def test_crawler_rejects_unsafe_url_forms(url):
    with pytest.raises(ValueError):
        WebCrawler._validate_public_url(url)


def test_gguf_context_limit_precedes_native_generation(monkeypatch):
    import sys

    native = MagicMock()
    native.tokenize.return_value = list(range(4096))
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=lambda **kwargs: native))
    engine = GGUFEngine("test.gguf")
    with pytest.raises(HTTPException) as error:
        engine.generate("too long")
    assert error.value.status_code == 422
    native.create_chat_completion.assert_not_called()


def test_missing_explicit_gguf_path_fails_clearly(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.gguf"))
    with pytest.raises(FileNotFoundError, match="MODEL_PATH does not exist"):
        GGUFEngine.from_environment()
