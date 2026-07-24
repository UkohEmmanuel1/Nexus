# Fine-Tuning

Guide for supervised fine-tuning (SFT), parameter-efficient fine-tuning with LoRA/QLoRA, and alignment via Direct Preference Optimization (DPO).

## Supervised Fine-Tuning (`src/finetuning/sft.py`)

Fine-tune on instruction datasets to teach the model to follow instructions:

```python
from src.finetuning.sft import SFTTrainer

trainer = SFTTrainer(
    model,
    tokenizer,
    config={
        "lr": 2e-5,
        "batch_size": 4,
        "epochs": 3,
        "max_seq_length": 2048,
        "output_dir": "sft_checkpoints",
    }
)

trainer.train([
    {
        "instruction": "What is the capital of France?",
        "output": "The capital of France is Paris."
    },
    # ... more examples
])
```

### Chat Format Training
```python
trainer.train([
    {
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"}
        ]
    }
])
```

## LoRA / QLoRA (`src/finetuning/lora.py`)

Parameter-efficient fine-tuning that trains low-rank adapters:

```python
from src.finetuning.lora import LoRATrainer

trainer = LoRATrainer(
    model,
    tokenizer,
    config={
        "r": 16,
        "alpha": 32,
        "dropout": 0.05,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "lr": 1e-4,
        "batch_size": 4,
        "epochs": 3,
    }
)

trainer.train(dataset)
trainer.save("checkpoints/lora_adapter.pt")
```

### QLoRA (4-bit Quantized LoRA)
```python
trainer = LoRATrainer(
    model,
    tokenizer,
    use_qlora=True,
    config={
        "r": 16,
        "alpha": 32,
        "dropout": 0.05,
        "lr": 1e-4,
        "batch_size": 4,
        "epochs": 3,
    }
)

# Requires bitsandbytes
# pip install bitsandbytes
```

### Merge LoRA Weights
```python
trainer.merge_and_unload()
# Saves the full merged model weights
```

## DPO (Direct Preference Optimization) (`src/finetuning/dpo.py`)

Align the model to prefer chosen responses over rejected ones:

```python
from src.finetuning.dpo import DPOTrainer

trainer = DPOTrainer(
    model,           # Policy model
    tokenizer,
    ref_model=ref_model,  # Reference model (frozen)
    config={
        "beta": 0.1,         # KL penalty coefficient
        "lr": 1e-5,
        "batch_size": 4,
        "epochs": 1,
    }
)

trainer.train([
    {
        "prompt": "What is the capital of France?",
        "chosen": "The capital of France is Paris.",
        "rejected": "idk France is a country",
    },
    # ...
])
```

### DPO Loss
```
L_DPO = -E[ log sigmoid(beta * (log(pi(y_w|x)/pi_ref(y_w|x)) - log(pi(y_l|x)/pi_ref(y_l|x)))) ]
```

Where `y_w` = chosen, `y_l` = rejected, `beta` = KL penalty coefficient.

## Dataset Formats

### SFT
```json
[
  {
    "instruction": "What is 2+2?",
    "input": "",
    "output": "2 + 2 = 4"
  }
]
```

### Chat
```json
[
  {
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hi"},
      {"role": "assistant", "content": "Hello! How can I help?"}
    ]
  }
]
```

### DPO
```json
[
  {
    "prompt": "What is 15 * 37?",
    "chosen": "Let me calculate... 15 * 37 = 555.",
    "rejected": "I don't know."
  }
]
```

## Training Commands

```bash
# SFT
python -m src.finetuning.sft \
    --model checkpoints/model.pt \
    --data data/sft_dataset.json \
    --lr 2e-5 \
    --epochs 3

# LoRA
python -m src.finetuning.lora \
    --model checkpoints/model.pt \
    --data data/sft_dataset.json \
    --r 16 \
    --alpha 32 \
    --lr 1e-4

# DPO
python -m src.finetuning.dpo \
    --model checkpoints/sft_model.pt \
    --ref checkpoints/sft_model.pt \
    --data data/dpo_dataset.json \
    --beta 0.1 \
    --lr 1e-5 \
    --epochs 1
```

## Best Practices

1. **Start with SFT**: Always fine-tune on instruction data first before alignment.
2. **Use LoRA for limited compute**: LoRA trains 1–2% of parameters; QLoRA enables fine-tuning 7B models on 8GB GPUs.
3. **DPO after SFT**: DPO works best when the policy and reference models start from a well-fine-tuned SFT checkpoint.
4. **Data quality > quantity**: 1000 high-quality examples often outperform 10K noisy ones.
5. **Evaluate before/after**: Use held-out validation sets to measure improvement.
