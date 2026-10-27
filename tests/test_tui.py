"""Drive the real full-screen input loop without network or a physical terminal."""

import asyncio
import importlib
from unittest.mock import AsyncMock

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput


@pytest.fixture(params=["src.inference", "nexus_terminal"])
def tui(request, monkeypatch, tmp_path):
    base = request.param
    api = importlib.import_module(base + (".assistant_cli" if base == "src.inference" else ".cli"))
    module = importlib.import_module(base + ".tui")
    monkeypatch.setenv("VONK_CONFIG_DIR", str(tmp_path))
    for p in api.PROVIDERS:
        monkeypatch.delenv(f"{p.upper()}_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    return api, module


def test_fullscreen_select_local_model_send_and_exit(tui, monkeypatch):
    api, module = tui
    async def exercise():
        started = asyncio.Event()

        async def answer(**kwargs):
            started.set()
            return {"response": "Hello\x1b]52;c;secret\x07 world"}

        complete = AsyncMock(side_effect=answer)
        monkeypatch.setattr(api, "complete_byok", complete)
        with create_pipe_input() as pipe:
            ui = module.Workspace(api, input=pipe, output=DummyOutput())
            ui.app.pre_run_callables.append(lambda: pipe.send_text("/free\r\rhello\r"))

            async def driver():
                await started.wait()
                await ui.task
                pipe.send_text("\x04")

            driver_task = asyncio.create_task(driver())
            assert await asyncio.wait_for(ui.app.run_async(), 4) == 0
            await driver_task
            assert ui.provider == "ollama"
            assert "Hello world" in ui.transcript
            assert "secret" not in ui.transcript
            assert not ui.busy
            assert complete.call_args.kwargs["provider"] == "ollama"
            assert complete.call_args.kwargs["messages"][-1]["content"] == "hello"

    asyncio.run(exercise())


def test_fullscreen_keeps_unconnected_prompt_and_supports_cancellation(tui, monkeypatch):
    api, module = tui

    async def exercise():
        with create_pipe_input() as pipe:
            ui = module.Workspace(api, input=pipe, output=DummyOutput())
            ui.composer.text = "keep my prompt"
            ui.send()
            assert ui.composer.text == "keep my prompt"
            assert "/connect" in ui.notice
            ui.select("ollama", "qwen2.5-coder:1.5b")
            started = asyncio.Event()

            async def waiting(**kwargs):
                started.set()
                await asyncio.Event().wait()

            monkeypatch.setattr(api, "complete_byok", waiting)

            async def driver():
                await started.wait()
                pipe.send_text("\x1b")
                await ui.task
                pipe.send_text("\x04")

            ui.composer.text = "hello"
            task = asyncio.create_task(driver())
            ui.app.pre_run_callables.append(ui.send)
            assert await asyncio.wait_for(ui.app.run_async(), 4) == 0
            await task
            assert "Request cancelled" in ui.transcript
            assert not ui.messages[1:]
            assert not ui.busy

    asyncio.run(exercise())
