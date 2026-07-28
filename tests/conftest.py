import pytest


@pytest.fixture
def mock_tokenizer():
    from unittest.mock import MagicMock

    from src.tokenizer import Tokenizer

    t = MagicMock(spec=Tokenizer)
    t.sp = None
    t.vocab_size = 100
    t.encode.return_value = [1, 2, 3]
    t.decode.return_value = "hello"
    t.is_loaded.return_value = False
    return t
