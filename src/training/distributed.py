import logging
import os

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def setup_distributed(backend: str = "nccl"):
    if not dist.is_available() or not torch.cuda.is_available():
        return False

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        dist.init_process_group(backend=backend)
        torch.cuda.set_device(local_rank)
        logger.info(
            f"Distributed initialized: rank={rank}, world_size={world_size}, local_rank={local_rank}"
        )
        return True
    return False


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    return get_rank() == 0


def wrap_fsdp(model: torch.nn.Module, **kwargs):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    from src.model.transformer_block import TransformerBlock

    auto_wrap_policy = transformer_auto_wrap_policy(transformer_layer_cls={TransformerBlock})
    return FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        **kwargs,
    )


def wrap_deepspeed(model: torch.nn.Module, config_path: str):
    import deepspeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=config_path,
    )
    return model_engine, optimizer
