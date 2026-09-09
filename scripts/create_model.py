import torch
import sys
sys.path.insert(0, '..')
from src.model import ModelConfig, Transformer
from src.model.config import MoEConfig
from src.utils.checkpoint import save_checkpoint

config = ModelConfig(
    dim=1536,
    n_layers=24,
    n_heads=12,
    n_kv_heads=2,
    ffn_mult=4912 / 1536,
    vocab_size=151936,
    max_seq_len=4096,
    norm_eps=1e-6,
    rope_theta=10000.0,
    hidden_dropout=0.0,
    attention_dropout=0.0,
    moe=MoEConfig(enabled=False),
    use_flash_attn=False,
    dtype="float32",
)

model = Transformer(config)
n_params = sum(p.numel() for p in model.parameters())
print(f"Created model with {n_params:,} parameters")

torch.save({"model_state_dict": model.state_dict()}, "checkpoints/model.pt")
print("Saved checkpoints/model.pt")
