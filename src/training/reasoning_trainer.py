from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.training.trainer import Trainer
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ReasoningTrainer(Trainer):
    def __init__(self, model: nn.Module, config: dict, **kwargs):
        super().__init__(model, config, **kwargs)
        self.thinking_loss_weight = config.get("thinking_loss_weight", 0.3)
        self.reward_temperature = config.get("reward_temperature", 0.1)

    def train_step(self, batch: dict) -> dict:
        input_ids = batch["input_ids"].to(self.device)
        labels = batch.get("labels", input_ids).to(self.device)
        thinking_mask = batch.get("thinking_mask", None)
        if thinking_mask is not None:
            thinking_mask = thinking_mask.to(self.device)

        with torch.amp.autocast(device_type="cuda", enabled=self.use_amp, dtype=self.amp_dtype):
            logits, aux_loss = self.model(input_ids)

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="none",
            )

            ce_loss = ce_loss.view(input_ids.shape[0], -1)

            if thinking_mask is not None:
                shift_thinking = thinking_mask[..., 1:].contiguous()
                thinking_loss = (ce_loss * shift_thinking.float()).sum() / (shift_thinking.float().sum() + 1e-8)
                standard_loss = (ce_loss * (1 - shift_thinking.float())).sum() / ((1 - shift_thinking.float()).sum() + 1e-8)
                loss = (1 - self.thinking_loss_weight) * standard_loss + self.thinking_loss_weight * thinking_loss
            else:
                loss = ce_loss.mean()

            loss = loss + aux_loss

        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return {"loss": loss.item(), "ce_loss": ce_loss.mean().item()}


class RLTrainer:
    def __init__(self, model: nn.Module, reward_fn: Callable, config: dict):
        self.model = model
        self.reward_fn = reward_fn
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = config.get("lr", 1e-6)
        self.clip_eps = config.get("clip_eps", 0.2)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)

    def compute_advantages(self, rewards: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        advantages = rewards - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    def train_on_batch(self, batch: dict) -> dict:
        old_log_probs = batch["old_log_probs"].to(self.device)
        advantages = batch["advantages"].to(self.device)
        input_ids = batch["input_ids"].to(self.device)

        logits, _ = self.model(input_ids)
        log_probs = F.log_softmax(logits, dim=-1)

        gathered = log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
        ratio = torch.exp(gathered - old_log_probs)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        self.optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {"policy_loss": policy_loss.item()}


def train_reasoning_sft(
    model,
    dataset,
    output_dir: str = "./checkpoints/reasoning",
    lr: float = 2e-5,
    num_epochs: int = 3,
    batch_size: int = 4,
    thinking_loss_weight: float = 0.3,
):
    config = {
        "lr": lr,
        "batch_size": batch_size,
        "max_steps": len(dataset) * num_epochs // batch_size,
        "warmup_steps": 100,
        "output_dir": output_dir,
        "dtype": "bfloat16",
        "thinking_loss_weight": thinking_loss_weight,
    }

    trainer = ReasoningTrainer(model, config)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    trainer.train(train_loader)
    return trainer
