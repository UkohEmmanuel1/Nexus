import pytest
import tempfile
from pathlib import Path

from src.tokenizer import Tokenizer


def test_tokenizer_no_model():
    tokenizer = Tokenizer("nonexistent.model")
    assert tokenizer.is_loaded() is False
    assert len(tokenizer) == 128000


def test_tokenizer_encode_without_model():
    tokenizer = Tokenizer("nonexistent.model")
    with pytest.raises(RuntimeError):
        tokenizer.encode("hello")


def test_tokenizer_special_ids():
    tokenizer = Tokenizer("nonexistent.model")
    assert isinstance(tokenizer.bos_id, int)
    assert isinstance(tokenizer.eos_id, int)
    assert isinstance(tokenizer.unk_id, int)
    assert isinstance(tokenizer.pad_id, int)


if __name__ == "__main__":
    pytest.main([__file__])
