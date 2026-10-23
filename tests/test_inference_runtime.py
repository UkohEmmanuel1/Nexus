from unittest.mock import MagicMock

import pytest
import torch

from src.inference.engine import InferenceEngine
from src.model import ModelConfig, Transformer
from src.model.config import LongContextConfig


def make_engine():
    model = Transformer(
        ModelConfig(
            dim=16,
            n_layers=1,
            n_heads=2,
            n_kv_heads=1,
            vocab_size=32,
            max_seq_len=64,
            long_context=LongContextConfig(enabled=False),
            use_flash_attn=False,
        )
    )
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3]
    tokenizer.decode.side_effect = lambda ids: " ".join(map(str, ids))
    tokenizer.eos_id = -1
    return InferenceEngine(model, tokenizer, device="cpu")


def test_cached_generation_matches_full_prefix(monkeypatch):
    engine = make_engine()
    original = engine.model.forward
    positions = []

    def forward(ids, start_pos=0, **kwargs):
        positions.append(start_pos)
        logits, aux = original(ids, start_pos=start_pos, **kwargs)
        prefix = torch.tensor([[1, 2, 3] + [4] * (len(positions) - 1)])
        expected, _ = original(prefix, use_cache=False)
        torch.testing.assert_close(logits[:, -1], expected[:, -1], atol=1e-5, rtol=1e-5)
        return logits, aux

    monkeypatch.setattr(engine.model, "forward", forward)
    monkeypatch.setattr(torch, "multinomial", lambda *args, **kwargs: torch.tensor([[4]]))
    output = engine.generate("prompt", max_new_tokens=3)
    assert output == "4 4 4"
    assert positions == [0, 3, 4]
    assert engine.model.layers[0].attention.cache_k is None


def test_closed_stream_cleans_cache_and_disables_gradients(monkeypatch):
    engine = make_engine()
    original = engine.model.forward

    def forward(*args, **kwargs):
        assert not torch.is_grad_enabled()
        return original(*args, **kwargs)

    monkeypatch.setattr(engine.model, "forward", forward)
    monkeypatch.setattr(torch, "multinomial", lambda *args, **kwargs: torch.tensor([[4]]))
    stream = engine.generate("prompt", max_new_tokens=3, stream=True)
    assert next(stream) == "4"
    stream.close()
    assert engine.model.layers[0].attention.cache_k is None


def test_thinking_does_not_fall_through_to_second_generation(monkeypatch):
    engine = make_engine()
    thinking = MagicMock(return_value=(torch.tensor([[1, 2, 3, 4]]), "trace"))
    regular = MagicMock(side_effect=AssertionError("generated twice"))
    monkeypatch.setattr(engine.model, "generate_with_thinking", thinking)
    monkeypatch.setattr(engine.model, "generate", regular)
    assert engine.generate("prompt", thinking_mode=True) == "4"


def test_missing_checkpoint_does_not_allocate_model(monkeypatch, tmp_path):
    from src.inference.cli import load_model

    constructor = MagicMock(side_effect=AssertionError("allocated before checking files"))
    monkeypatch.setattr("src.inference.cli.Transformer", constructor)
    with pytest.raises(FileNotFoundError, match="Model checkpoint not found"):
        load_model(str(tmp_path / "missing.pt"), "missing.model", None, "cpu")
    constructor.assert_not_called()
