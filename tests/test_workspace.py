import importlib
import json
import pathlib
import sys
from unittest.mock import Mock

import pytest


@pytest.fixture(params=["src.inference", "nexus_terminal"])
def workspace(request, tmp_path, monkeypatch):
    base = request.param
    cli = importlib.import_module(base + (".assistant_cli" if base == "src.inference" else ".cli"))
    module = importlib.import_module(base + ".workspace")
    monkeypatch.setenv("VONK_CONFIG_DIR", str(tmp_path / "config"))
    for provider in cli.PROVIDERS:
        monkeypatch.delenv(f"{provider.upper()}_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    return cli, module, tmp_path


@pytest.mark.parametrize("argv", [[], ["chat"], ["ui"]])
def test_workspace_opens_without_a_key_and_stays_open_after_chat_selected(
    workspace, monkeypatch, capsys, argv
):
    cli, _, _ = workspace
    answers = iter(["1", "/help", "0"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert cli.main(argv) == 0
    output = capsys.readouterr().out
    for text in (
        "V O N K",
        "Select model",
        "Connect provider",
        "Free models",
        "Skills",
        "MCP",
        "CONNECT BEFORE CHATTING",
    ):
        assert text in output
    assert "No key for deepseek" not in output


def test_workspace_selects_a_free_local_model_without_key(workspace, monkeypatch):
    cli, _, _ = workspace
    answers = iter(["4", "1", "0"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert cli.main(["ui"]) == 0
    settings = cli.load(cli.config_paths()[0])
    assert settings == {"provider": "ollama", "model": "qwen2.5-coder:1.5b"}
    assert cli.credential("ollama", cli.config_paths()[1])[1] == "local / no API key"


def test_skills_are_used_only_when_enabled(workspace, monkeypatch):
    cli, _, directory = workspace
    source = directory / "SKILL.md"
    source.write_text("Always explain test coverage.", encoding="utf-8")
    assert cli.main(["skills", "add", str(source), "--name", "review"]) == 0
    monkeypatch.setenv("HF_TOKEN", "test_key")
    generate = Mock(return_value="done")
    monkeypatch.setattr(cli, "request_chat", generate)
    assert cli.main(["chat", "--provider", "huggingface", "review code"]) == 0
    assert "Always explain test coverage." in generate.call_args.args[2][0]["content"]
    assert cli.main(["skills", "disable", "review"]) == 0
    assert cli.main(["chat", "--provider", "huggingface", "review code"]) == 0
    assert "Always explain test coverage." not in generate.call_args.args[2][0]["content"]


def test_mcp_real_subprocess_handshake_tool_listing_and_call(workspace, monkeypatch, capsys):
    cli, _, _ = workspace
    server = pathlib.Path(__file__).parent / "fixtures" / "mcp_echo_server.py"
    assert cli.main(["mcp", "add", "echo", sys.executable, str(server)]) == 0
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    assert cli.main(["mcp", "tools", "echo"]) == 2
    assert cli.main(["mcp", "tools", "echo", "--yes"]) == 0
    assert "Echo test input" in capsys.readouterr().out
    monkeypatch.setenv("HF_TOKEN", "synthetic-secret-not-for-mcp")
    assert (
        cli.main(["mcp", "call", "echo", "echo", json.dumps({"text": "MCP works"}), "--yes"]) == 0
    )
    output = capsys.readouterr().out
    assert "MCP works" in output and '"credential_leaked": false' in output


def test_extension_paths_and_sizes_are_validated(workspace):
    cli, module, directory = workspace
    store = module.Store(cli.config_paths()[0].parent)
    with pytest.raises(ValueError):
        store.add_mcp("../escape", "python", [])
    source = directory / "SKILL.md"
    source.write_bytes(b"x" * 32769)
    assert cli.main(["skills", "add", str(source), "--name", "large"]) == 2
    with pytest.raises(ValueError):
        store.add_mcp("bad", "cmd.cmd", [])
