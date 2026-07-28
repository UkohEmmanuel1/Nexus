
import torch

from src.utils.logging import get_logger

logger = get_logger(__name__)


def apply_lora(
    model: torch.nn.Module,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: list[str] = None,
) -> torch.nn.Module:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError:
        logger.error("peft not installed. Install with: pip install peft")
        raise

    if target_modules is None:
        target_modules = ["wq", "wk", "wv", "wo", "w1", "w2", "w3"]

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def apply_qlora(
    model: torch.nn.Module,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    bnb_4bit_compute_dtype: torch.dtype = torch.bfloat16,
    bnb_4bit_quant_type: str = "nf4",
    use_double_quant: bool = True,
) -> torch.nn.Module:
    try:
        import bitsandbytes as bnb
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import BitsAndBytesConfig
    except ImportError:
        logger.error("bitsandbytes or peft not installed")
        raise

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=use_double_quant,
    )

    logger.warning("QLoRA requires model to be loaded with quantization config")

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["wq", "wk", "wv", "wo", "w1", "w2", "w3"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model
