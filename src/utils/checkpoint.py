from pathlib import Path

import torch

from .logging import get_logger

logger = get_logger(__name__)


def save_checkpoint(
    state: dict,
    filepath: str,
    keep_last_n: int = 3,
) -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(path))
    logger.info(f"Checkpoint saved to {filepath}")

    sibling_files = sorted(path.parent.glob(f"{path.stem}*{path.suffix}"))
    while len(sibling_files) > keep_last_n:
        oldest = sibling_files.pop(0)
        oldest.unlink()
        logger.info(f"Removed old checkpoint: {oldest}")


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module = None,
    optimizer: torch.optim.Optimizer = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    map_location: str = None,
    strict: bool = True,
) -> dict:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    logger.info(f"Loading checkpoint from {filepath}")
    state = torch.load(str(path), map_location=map_location, weights_only=True)

    if model is not None and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=strict)
        logger.info("Model weights loaded")

    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
        logger.info("Optimizer state loaded")

    if scheduler is not None and "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])
        logger.info("Scheduler state loaded")

    return state
