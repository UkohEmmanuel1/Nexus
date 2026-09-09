import torch
from torch.utils.data import DataLoader

from src.training.trainer import Trainer
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SFTTrainer(Trainer):
    def train_step(self, batch: dict) -> dict:
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch.get("attention_mask", None)
        labels = batch.get("labels", input_ids).to(self.device)

        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.amp.autocast(device_type="cuda", enabled=self.use_amp, dtype=self.amp_dtype):
            logits, _ = self.model(input_ids)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return {"loss": loss.item()}


def finetune_sft(
    model,
    train_dataset,
    val_dataset=None,
    output_dir: str = "./checkpoints/sft",
    lr: float = 2e-5,
    num_epochs: int = 3,
    batch_size: int = 4,
    use_wandb: bool = False,
    resume_from: str | None = None,
):
    config = {
        "lr": lr,
        "batch_size": batch_size,
        "max_steps": len(train_dataset) * num_epochs // batch_size,
        "warmup_steps": 100,
        "output_dir": output_dir,
        "dtype": "bfloat16",
        "log_interval": 10,
        "save_interval": 500,
        "eval_interval": 200,
    }

    trainer = SFTTrainer(model, config, use_wandb=use_wandb)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        if val_dataset
        else None
    )

    trainer.train(train_loader, eval_dataloader=val_loader, resume_from=resume_from)
    return trainer
