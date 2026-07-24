# Training

Instructions for pretraining the model from scratch or continuing training from a checkpoint. Supports single-GPU, multi-GPU FSDP, and DeepSpeed ZeRO.

## Training Loop (`src/training/trainer.py`)

```python
from src.training.trainer import Trainer
from src.model import Transformer, ModelConfig
from src.data import DataPipeline

# Initialize
model = Transformer(ModelConfig.from_yaml("config/model/model_3b.yaml"))
trainer = Trainer(
    model,
    config={
        "lr": 3e-4,
        "weight_decay": 0.1,
        "betas": (0.9, 0.95),
        "max_lr": 3e-4,
        "min_lr": 3e-5,
        "warmup_steps": 2000,
        "total_steps": 500000,
        "batch_size": 8,
        "gradient_accumulation_steps": 16,
        "max_grad_norm": 1.0,
        "save_every_steps": 5000,
        "eval_every_steps": 1000,
        "eval_steps": 100,
        "log_every_steps": 10,
        "output_dir": "checkpoints",
        "resume_from_checkpoint": None,
        "dtype": "bfloat16",
    }
)

# Train
trainer.train(train_dataloader, eval_dataloader)
```

## Distributed Training (`src/training/distributed.py`)

### FSDP (Fully Sharded Data Parallel)
```bash
torchrun --nproc_per_node=4 src/training/trainer.py \
    --config config/training/fsdp.yaml \
    --model config/model/model_3b.yaml
```

`config/training/fsdp.yaml`:
```yaml
training:
  distributed_strategy: fsdp
  fsdp:
    sharding_strategy: HYBRID_SHARD
    auto_wrap_policy: transformer_auto_wrap_policy
    mixed_precision: bf16
    cpu_offload: false
    backward_prefetch: BACKWARD_PRE
    limit_all_gathers: true
```

### DeepSpeed ZeRO
```bash
torchrun --nproc_per_node=8 src/training/trainer.py \
    --config config/training/deepspeed.yaml \
    --model config/model/model_3b.yaml
```

`config/training/deepspeed.yaml`:
```yaml
training:
  distributed_strategy: deepspeed
  deepspeed:
    zero_optimization:
      stage: 2
      offload_param: none
      offload_optimizer: none
    gradient_accumulation_steps: 16
    gradient_clipping: 1.0
    fp16:
      enabled: false
    bf16:
      enabled: true
```

## Configurations

| Model | GPUs | Batch/GPU | Grad Accum | Effective BS | LR | Warmup |
|-------|------|-----------|------------|-------------|----|--------|
| 1B | 1 | 8 | 32 | 256 | 3e-4 | 2000 |
| 3B | 4 | 4 | 16 | 256 | 2e-4 | 2000 |
| 7B | 8 | 2 | 16 | 256 | 1.5e-4 | 1000 |
| 13B | 16 | 1 | 16 | 256 | 1e-4 | 1000 |
| 32B | 32 | 1 | 8 | 256 | 8e-5 | 500 |
| 70B | 64 | 1 | 4 | 256 | 5e-5 | 500 |

## Logging

Logs are written to `logs/train_{timestamp}.log` with:

```
[INFO] Step 1000 | Loss: 3.245 | LR: 2.5e-4 | Tokens/s: 45000 | Grad norm: 0.89 | Throughput: 45K tok/s/GPU
```

Weights & Biases logging is optional (`use_wandb: false` in config).

## Checkpoints

Saved to `checkpoints/step_{step}/`:
```
checkpoints/
├── step_5000/
│   ├── model.pt          # Model weights
│   ├── optimizer.pt      # Optimizer state
│   ├── config.yaml       # Training config
│   └── training_state.pt # Step, epoch, LR
└── step_10000/
    └── ...
```

Resume from checkpoint:
```python
trainer = Trainer(model, config={"resume_from_checkpoint": "checkpoints/step_5000"})
```

## Dataset Format

Expects tokenized `.pt` files:
```python
# Each file contains torch.save({
#     "input_ids": torch.tensor([...]),   # (seq_len,)
#     "attention_mask": torch.tensor([...])  # (seq_len,)
# })
```

Data pipeline produces these via:
```bash
python -m src.data.tokenize_dataset \
    --input data/raw \
    --output data/tokenized \
    --tokenizer tokenizer/tokenizer.model \
    --max_seq_len 4096
```
