import logging
import math
import time
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.utils.logging import get_logger

logger = get_logger(__name__)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        use_wandb: bool = False,
    ):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_wandb = use_wandb

        self.grad_accum_steps = config.get("grad_accum_steps", 1)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self.log_interval = config.get("log_interval", 10)
        self.save_interval = config.get("save_interval", 1000)
        self.eval_interval = config.get("eval_interval", 500)
        self.max_steps = config.get("max_steps", 100000)
        self.warmup_steps = config.get("warmup_steps", 1000)
        self.lr = config.get("lr", 3e-4)
        self.weight_decay = config.get("weight_decay", 0.1)
        self.beta1 = config.get("beta1", 0.9)
        self.beta2 = config.get("beta2", 0.95)
        self.eps = config.get("eps", 1e-8)
        self.output_dir = Path(config.get("output_dir", "./checkpoints"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(self.beta1, self.beta2),
            eps=self.eps,
        )

        self._build_scheduler()

        self.scaler = torch.amp.GradScaler("cuda", enabled=(config.get("dtype", "bfloat16") == "float16"))
        self.use_amp = config.get("dtype", "bfloat16") in ("float16", "bfloat16")
        self.amp_dtype = torch.bfloat16 if config.get("dtype") == "bfloat16" else torch.float16

        self.global_step = 0
        self.epoch = 0
        self.best_loss = float("inf")

        model.to(self.device)

    def _build_scheduler(self):
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.max_steps - self.warmup_steps,
            eta_min=self.lr * 0.1,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_steps],
        )

    def train_step(self, batch: Dict) -> Dict:
        input_ids = batch["input_ids"].to(self.device)
        labels = batch.get("labels", input_ids).to(self.device)

        with torch.amp.autocast(device_type="cuda", enabled=self.use_amp, dtype=self.amp_dtype):
            logits, aux_loss = self.model(input_ids)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=0,
            )
            loss = loss + aux_loss

        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return {"loss": loss.item()}

    def _optimizer_step(self):
        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        if self.scaler.is_enabled():
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()

    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        resume_from: Optional[str] = None,
    ):
        if resume_from:
            self._resume(resume_from)

        logger.info(f"Starting training on {self.device}")
        logger.info(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")

        self.model.train()
        running_loss = 0.0
        accum_loss = 0.0
        start_time = time.time()

        while self.global_step < self.max_steps:
            for batch in train_dataloader:
                step_loss = self.train_step(batch)
                running_loss += step_loss["loss"]
                accum_loss += step_loss["loss"]

                if (self.global_step + 1) % self.grad_accum_steps == 0:
                    self._optimizer_step()

                if self.global_step % self.log_interval == 0 and self.global_step > 0:
                    avg_loss = running_loss / self.log_interval
                    lr = self.scheduler.get_last_lr()[0]
                    ms_per_step = (time.time() - start_time) / self.log_interval * 1000

                    logger.info(
                        f"Step {self.global_step}/{self.max_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {lr:.2e} | "
                        f"Speed: {ms_per_step:.1f}ms/step"
                    )

                    if self.use_wandb:
                        import wandb
                        wandb.log({
                            "loss": avg_loss,
                            "lr": lr,
                            "perplexity": math.exp(avg_loss),
                            "step": self.global_step,
                        })

                    running_loss = 0.0
                    start_time = time.time()

                if self.global_step % self.save_interval == 0 and self.global_step > 0:
                    self._save()

                if eval_dataloader and self.global_step % self.eval_interval == 0 and self.global_step > 0:
                    eval_loss = self.evaluate(eval_dataloader)
                    logger.info(f"Eval loss: {eval_loss:.4f}, perplexity: {math.exp(eval_loss):.2f}")
                    if eval_loss < self.best_loss:
                        self.best_loss = eval_loss
                        self._save(best=True)
                    self.model.train()

                self.global_step += 1
                if self.global_step >= self.max_steps:
                    break

        self._save(final=True)
        logger.info("Training complete")

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch.get("labels", input_ids).to(self.device)

            logits, _ = self.model(input_ids)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=0,
            )
            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def _save(self, best: bool = False, final: bool = False):
        suffix = "best" if best else ("final" if final else f"step_{self.global_step}")
        path = self.output_dir / f"checkpoint_{suffix}.pt"

        state = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "config": self.config,
        }
        save_checkpoint(state, str(path))

    def _resume(self, filepath: str):
        state = load_checkpoint(
            filepath,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )
        self.global_step = state.get("global_step", 0)
        self.epoch = state.get("epoch", 0)
        self.best_loss = state.get("best_loss", float("inf"))
        logger.info(f"Resumed from step {self.global_step}")


def main():
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../../configs/training", config_name="pretrain")
    def _main(cfg: DictConfig):
        from src.model import Transformer, ModelConfig, MoEConfig
        model_cfg = ModelConfig(
            dim=cfg.model.dim,
            n_layers=cfg.model.n_layers,
            n_heads=cfg.model.n_heads,
            n_kv_heads=cfg.model.n_kv_heads,
            ffn_mult=cfg.model.ffn_mult,
            vocab_size=cfg.model.vocab_size,
            max_seq_len=cfg.model.max_seq_len,
            norm_eps=cfg.model.norm_eps,
            rope_theta=cfg.model.rope_theta,
            moe=MoEConfig(
                enabled=cfg.model.moe.enabled,
                num_experts=cfg.model.moe.num_experts,
                top_k=cfg.model.moe.top_k,
            ),
            use_flash_attn=cfg.model.use_flash_attn,
            dtype=cfg.model.dtype,
        )
        model = Transformer(model_cfg)
        trainer = Trainer(model, dict(cfg.training), use_wandb=cfg.training.get("wandb", False))
        dataset = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.randint(0, model_cfg.vocab_size, (100, model_cfg.max_seq_len))
            ),
            batch_size=cfg.training.batch_size,
        )
        trainer.train(dataset)

    _main()


if __name__ == "__main__":
    main()
