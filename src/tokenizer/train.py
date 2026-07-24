import argparse
import logging
from pathlib import Path

import sentencepiece as spm

logger = logging.getLogger(__name__)

DEFAULT_SPECIAL_TOKENS = [
    "<unk>", "<s>", "</s>", "<pad>",
    "<mask>", "<sep>", "<cls>",
    "<tool_call>", "<tool_result>", "<function_call>", "<function_result>",
    "<|im_start|>", "<|im_end|>",
    "<|system|>", "<|user|>", "<|assistant|>",
]


def train_tokenizer(
    input_files: list[str],
    model_prefix: str = "tokenizer",
    vocab_size: int = 128000,
    model_type: str = "bpe",
    character_coverage: float = 1.0,
    max_sentencepiece_length: int = 16,
    num_threads: int = 8,
    train_extremely_large_corpus: bool = True,
    split_by_unicode_script: bool = True,
    split_by_number: bool = True,
    split_by_whitespace: bool = True,
    byte_fallback: bool = True,
    user_defined_symbols: list[str] = None,
):
    if user_defined_symbols is None:
        user_defined_symbols = DEFAULT_SPECIAL_TOKENS

    params = dict(
        input=",".join(input_files),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        max_sentencepiece_length=max_sentencepiece_length,
        num_threads=num_threads,
        train_extremely_large_corpus=train_extremely_large_corpus,
        split_by_unicode_script=split_by_unicode_script,
        split_by_number=split_by_number,
        split_by_whitespace=split_by_whitespace,
        byte_fallback=byte_fallback,
        user_defined_symbols=user_defined_symbols,
        pad_id=3,
        pad_piece="<pad>",
        unk_id=0,
        unk_piece="<unk>",
        bos_id=1,
        bos_piece="<s>",
        eos_id=2,
        eos_piece="</s>",
    )

    logger.info(f"Training tokenizer with vocab_size={vocab_size}, model_type={model_type}")
    logger.info(f"Input files: {input_files}")
    logger.info(f"Special tokens: {user_defined_symbols}")

    spm.SentencePieceTrainer.train(**params)
    logger.info(f"Tokenizer saved to {model_prefix}.model and {model_prefix}.vocab")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, nargs="+", required=True)
    parser.add_argument("--model-prefix", type=str, default="tokenizer")
    parser.add_argument("--vocab-size", type=int, default=128000)
    parser.add_argument("--model-type", type=str, default="bpe")
    parser.add_argument("--character-coverage", type=float, default=1.0)
    args = parser.parse_args()

    train_tokenizer(
        input_files=args.input,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
    )
