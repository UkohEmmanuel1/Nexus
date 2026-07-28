from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.trainer import Trainer
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DPOConfig:
    beta: float = 0.1
    label_smoothing: float = 0.0
    loss_type: str = "sigmoid"


class DPOTrainer(Trainer):
    def __init__(self, model: nn.Module, ref_model: nn.Module, config: dict, dpo_config: DPOConfig | None = None):
        super().__init__(model, config)
        self.ref_model = ref_model
        self.ref_model.to(self.device)
        self.ref_model.eval()
        self.dpo_config = dpo_config or DPOConfig()

    def train_step(self, batch: dict) -> dict:
        chosen_ids = batch["chosen_input_ids"].to(self.device)
        rejected_ids = batch["rejected_input_ids"].to(self.device)

        with torch.amp.autocast(device_type="cuda", enabled=self.use_amp, dtype=self.amp_dtype):
            chosen_logits, _ = self.model(chosen_ids)
            rejected_logits, _ = self.model(rejected_ids)

            with torch.no_grad():
                ref_chosen_logits, _ = self.ref_model(chosen_ids)
                ref_rejected_logits, _ = self.ref_model(rejected_ids)

            chosen_log_probs = self._log_probs_from_logits(chosen_logits, chosen_ids)
            rejected_log_probs = self._log_probs_from_logits(rejected_logits, rejected_ids)
            ref_chosen_log_probs = self._log_probs_from_logits(ref_chosen_logits, chosen_ids)
            ref_rejected_log_probs = self._log_probs_from_logits(ref_rejected_logits, rejected_ids)

            policy_chosen_log_probs = chosen_log_probs - ref_chosen_log_probs
            policy_rejected_log_probs = rejected_log_probs - ref_rejected_log_probs

            logits = self.dpo_config.beta * (policy_chosen_log_probs - policy_rejected_log_probs)

            if self.dpo_config.loss_type == "sigmoid":
                loss = -F.logsigmoid(logits).mean()
            elif self.dpo_config.loss_type == "hinge":
                loss = torch.relu(1 - logits).mean()
            else:
                raise ValueError(f"Unknown loss type: {self.dpo_config.loss_type}")

        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return {"loss": loss.item()}

    def _log_probs_from_logits(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        log_probs = F.log_softmax(shift_logits, dim=-1)
        per_token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        return per_token_log_probs.sum(-1) / (shift_labels != -100).sum(-1).float()
