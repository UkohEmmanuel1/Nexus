import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.model import ModelConfig, Transformer
from src.training.trainer import Trainer


def get_test_config() -> dict:
    return {
        "lr": 1e-4,
        "weight_decay": 0.01,
        "beta1": 0.9,
        "beta2": 0.95,
        "eps": 1e-8,
        "max_steps": 5,
        "warmup_steps": 1,
        "batch_size": 2,
        "grad_accum_steps": 1,
        "max_grad_norm": 1.0,
        "dtype": "float32",
        "log_interval": 1,
        "save_interval": 100,
        "eval_interval": 100,
        "output_dir": "./tests/tmp_checkpoints",
    }


def test_trainer_creation():
    config = get_test_config()
    model_config = ModelConfig(
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=100,
        max_seq_len=32,
        use_flash_attn=False,
        dtype="float32",
    )
    model = Transformer(model_config)
    trainer = Trainer(model, config)

    assert trainer is not None
    assert trainer.global_step == 0


def test_trainer_training_step():
    config = get_test_config()
    model_config = ModelConfig(
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=100,
        max_seq_len=32,
        use_flash_attn=False,
        dtype="float32",
    )
    model = Transformer(model_config)
    trainer = Trainer(model, config)

    dataset = TensorDataset(
        torch.randint(0, 100, (10, 32)),
        torch.randint(0, 100, (10, 32)),
    )
    loader = DataLoader(dataset, batch_size=2)

    batch = next(iter(loader))
    result = trainer.train_step({"input_ids": batch[0], "labels": batch[1]})

    assert "loss" in result
    assert result["loss"] > 0


class DictDataset(torch.utils.data.Dataset):
    def __init__(self, size=10, seq_len=32, vocab=100):
        self.size = size
        self.seq_len = seq_len
        self.vocab = vocab

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        ids = torch.randint(0, self.vocab, (self.seq_len,))
        return {"input_ids": ids, "labels": ids}


def test_trainer_evaluate():
    config = get_test_config()
    model_config = ModelConfig(
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=100,
        max_seq_len=32,
        use_flash_attn=False,
        dtype="float32",
    )
    model = Transformer(model_config)
    trainer = Trainer(model, config)

    dataset = DictDataset()
    loader = DataLoader(dataset, batch_size=2)

    loss = trainer.evaluate(loader)
    assert loss > 0


if __name__ == "__main__":
    pytest.main([__file__])
