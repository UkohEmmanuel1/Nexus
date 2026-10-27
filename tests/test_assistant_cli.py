"""Exercise both Python distributions through their real CLI entry points."""

import importlib
import json
from unittest.mock import Mock

import httpx
import pytest


@pytest.fixture(params=["src.inference.assistant_cli", "nexus_terminal.cli"])
def cli(request, monkeypatch, tmp_path):
    module = importlib.import_module(request.param)
    monkeypatch.setenv("VONK_CONFIG_DIR", str(tmp_path / "config"))
    for provider in module.PROVIDERS:
        monkeypatch.delenv(f"{provider.upper()}_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    return module


def test_models_and_selected_aliases(cli, capsys):
    assert cli.main(["models", "huggingface"]) == 0
    output = capsys.readouterr().out
    for name in ("qwen", "deepseek", "mimo"):
        assert name in output
    assert cli.main(["use", "huggingface", "mimo"]) == 0
    config, _ = cli.config_paths()
    assert cli.load(config)["model"] == "XiaomiMiMo/MiMo-V2.5"
    assert cli.main(["use", "opencode"]) == 0
    assert cli.load(config)["model"] == "big-pickle"


def test_missing_key_and_invalid_config_fail_without_traceback(cli, capsys):
    assert cli.main(["chat", "hello"]) == 2
    assert "No key" in capsys.readouterr().err
    config, _ = cli.config_paths()
    config.parent.mkdir(parents=True)
    config.write_text("[]", encoding="utf-8")
    assert cli.main(["status"]) == 2
    assert "JSON object" in capsys.readouterr().err
    config.write_text("broken JSON", encoding="utf-8")
    assert cli.main(["status"]) == 2
    assert "Traceback" not in capsys.readouterr().err


def test_provider_override_uses_that_providers_model(cli, monkeypatch):
    assert cli.main(["use", "deepseek", "deepseek-chat"]) == 0
    monkeypatch.setenv("HF_TOKEN", "hf_test_key")
    generate = Mock(return_value="done")
    monkeypatch.setattr(cli, "request_chat", generate)
    assert cli.main(["chat", "--provider", "huggingface", "hello"]) == 0
    assert generate.call_args.args[:2] == ("huggingface", "Qwen/Qwen2.5-Coder-32B-Instruct")


@pytest.mark.parametrize(
    ("provider", "alias", "expected", "key_name"),
    [
        ("huggingface", "qwen", "Qwen/Qwen2.5-Coder-32B-Instruct", "HF_TOKEN"),
        ("huggingface", "deepseek", "deepseek-ai/DeepSeek-V3.2", "HF_TOKEN"),
        ("huggingface", "mimo", "XiaomiMiMo/MiMo-V2.5", "HF_TOKEN"),
        ("opencode", "big-pickle", "big-pickle", "OPENCODE_API_KEY"),
    ],
)
def test_real_cli_routes_each_requested_model(
    cli, monkeypatch, capsys, provider, alias, expected, key_name
):
    captured = []
    actual_client = httpx.AsyncClient

    def upstream(request):
        captured.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "Answer: 4"}}]})

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: actual_client(transport=httpx.MockTransport(upstream), **kwargs),
    )
    monkeypatch.setenv(key_name, "synthetic_test_credential")
    assert cli.main(["chat", "--provider", provider, "--model", alias, "What is 2 + 2?"]) == 0
    assert json.loads(captured[0].content)["model"] == expected
    assert str(captured[0].url) == cli.PROVIDERS[provider][0]
    assert "Vonk: Answer: 4" in capsys.readouterr().out


@pytest.mark.parametrize("status", [401, 403, 404, 429, 500])
def test_provider_failures_exit_nonzero_without_exposing_upstream_body(
    cli, monkeypatch, capsys, status
):
    actual_client = httpx.AsyncClient
    transport = httpx.MockTransport(
        lambda request: httpx.Response(status, text="PRIVATE_KEY_OR_PROMPT")
    )
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: actual_client(transport=transport, **kw))
    monkeypatch.setenv("HF_TOKEN", "synthetic_test_credential")
    assert cli.main(["chat", "--provider", "huggingface", "--model", "qwen", "hello"]) == 1
    error = capsys.readouterr().err
    assert "PRIVATE_KEY_OR_PROMPT" not in error
    assert "Traceback" not in error


def test_interactive_history_clear_and_failed_turn_recovery(cli, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test_key")
    prompts = iter(["first", "failed", "next", "/clear", "fresh", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(prompts))
    turns = []

    def generate(provider, model, messages, *args):
        turns.append([dict(m) for m in messages])
        if messages[-1]["content"] == "failed":
            raise RuntimeError("secret upstream error")
        return "answer"

    monkeypatch.setattr(cli, "request_chat", generate)
    assert cli.main(["chat", "--provider", "huggingface"]) == 0
    assert [m["content"] for m in turns[2][1:]] == ["first", "answer", "next"]
    assert len(turns[3]) == 2


def test_credential_precedence_status_logout_and_hidden_login(cli, monkeypatch, capsys):
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli.getpass, "getpass", lambda _: "saved_secret")
    assert cli.main(["login", "huggingface"]) == 0
    assert cli.main(["login", "deepseek"]) == 0
    assert cli.main(["use", "huggingface", "qwen"]) == 0
    monkeypatch.setenv("HF_TOKEN", "environment_secret")
    assert cli.main(["status"]) == 0
    output = capsys.readouterr().out
    assert "HF_TOKEN" in output
    assert "saved_secret" not in output and "environment_secret" not in output
    assert cli.main(["logout", "huggingface"]) == 0
    _, path = cli.config_paths()
    assert cli.load(path) == {"deepseek": "saved_secret"}


def test_files_are_bounded_validated_and_never_modified(cli, tmp_path, monkeypatch):
    path = tmp_path / "example.py"
    path.write_text("print('hello')", encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN", "test_key")
    generate = Mock(return_value="Review complete")
    monkeypatch.setattr(cli, "request_chat", generate)
    assert cli.main(["chat", "--provider", "huggingface", "--file", str(path), "review"]) == 0
    assert "print('hello')" in generate.call_args.args[2][0]["content"]
    assert path.read_text() == "print('hello')"
    for raw in (b"x" * 60001, b"\x00binary", b"\xffbadutf8"):
        path.write_bytes(raw)
        assert cli.main(["chat", "--provider", "huggingface", "--file", str(path), "review"]) == 2
    assert cli.main(["chat", "--file", str(tmp_path / ".env"), "review"]) == 2


def test_noninteractive_login_and_terminal_control_safety(cli, monkeypatch):
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    assert cli.main(["login", "huggingface"]) == 2
    assert cli.terminal_text("\x1b[31mhello\x1b[0m\x1b]52;c;payload\x07\x9b") == "hello"


@pytest.mark.parametrize("args", [["--max-tokens", "0"], ["--temperature", "nan"]])
def test_invalid_generation_settings(cli, args):
    assert cli.main(["chat", *args, "hello"]) == 2
