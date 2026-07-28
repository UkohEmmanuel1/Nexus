import logging
from pathlib import Path

import sentencepiece as spm

logger = logging.getLogger(__name__)


class Tokenizer:
    def __init__(
        self,
        model_path: str,
        vocab_size: int = 128000,
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
    ):
        self.model_path = model_path
        self.vocab_size = vocab_size
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pad_token = pad_token

        self.sp = None
        if Path(model_path).exists():
            self.sp = spm.SentencePieceProcessor(model_file=str(model_path))
            logger.info(f"Loaded tokenizer from {model_path}")
            logger.info(f"Vocab size: {self.sp.vocab_size()}")
        else:
            logger.warning(f"Tokenizer model not found at {model_path}")

    @property
    def bos_id(self) -> int:
        return self.sp.bos_id() if self.sp else 1

    @property
    def eos_id(self) -> int:
        return self.sp.eos_id() if self.sp else 2

    @property
    def unk_id(self) -> int:
        return self.sp.unk_id() if self.sp else 0

    @property
    def pad_id(self) -> int:
        return self.sp.pad_id() if self.sp else 0

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> list[int]:
        if self.sp is None:
            raise RuntimeError("Tokenizer not loaded")
        return self.sp.encode(text, add_bos=add_bos, add_eos=add_eos)

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        if self.sp is None:
            raise RuntimeError("Tokenizer not loaded")
        return self.sp.decode(ids)

    def encode_batch(
        self, texts: list[str], add_bos: bool = True, add_eos: bool = False
    ) -> list[list[int]]:
        if self.sp is None:
            raise RuntimeError("Tokenizer not loaded")
        return self.sp.encode(texts, add_bos=add_bos, add_eos=add_eos)

    def decode_batch(self, batch: list[list[int]]) -> list[str]:
        if self.sp is None:
            raise RuntimeError("Tokenizer not loaded")
        return self.sp.decode(batch)

    def is_loaded(self) -> bool:
        return self.sp is not None

    def __len__(self) -> int:
        return self.sp.vocab_size() if self.sp else self.vocab_size

    def __call__(self, text: str, **kwargs) -> list[int]:
        return self.encode(text, **kwargs)
