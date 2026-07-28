import pytest
import torch

from src.model import ModelConfig, MoEConfig, Transformer


def get_test_config(moe: bool = False) -> ModelConfig:
    return ModelConfig(
        dim=128,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        ffn_mult=2.6667,
        vocab_size=1000,
        max_seq_len=128,
        norm_eps=1e-5,
        moe=MoEConfig(enabled=moe, num_experts=4, top_k=2),
        use_flash_attn=False,
        dtype="float32",
    )


def test_model_creation():
    config = get_test_config()
    model = Transformer(config)
    assert model is not None


def test_model_forward():
    config = get_test_config()
    model = Transformer(config)
    model.eval()

    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits, aux_loss = model(input_ids)

    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert isinstance(aux_loss, torch.Tensor)


def test_model_with_moe():
    config = get_test_config(moe=True)
    model = Transformer(config)
    model.eval()

    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits, aux_loss = model(input_ids)

    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert aux_loss.item() > 0


def test_model_generate():
    config = get_test_config()
    model = Transformer(config)
    model.eval()

    batch_size = 1
    seq_len = 8
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=16, temperature=1.0, top_k=10)

    assert output.shape[0] == batch_size
    assert output.shape[1] > seq_len


def test_model_training_step():
    config = get_test_config()
    model = Transformer(config)
    model.train()

    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()

    logits, aux_loss = model(input_ids)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    loss = loss + aux_loss
    loss.backward()

    assert loss.item() > 0
    for param in model.parameters():
        if param.grad is not None:
            assert param.grad is not None
            break


def test_kv_cache():
    config = get_test_config()
    model = Transformer(config)
    model.eval()

    batch_size, seq_len = 1, 8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    model.init_kv_cache(batch_size, config.max_seq_len, device, torch.float32)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    with torch.no_grad():
        logits1, _ = model(input_ids, use_cache=True)
        logits2, _ = model(input_ids[:, -1:], start_pos=seq_len, use_cache=True)

    assert logits1 is not None
    assert logits2 is not None

    model.reset_kv_cache()
    assert model.layers[0].attention.cache_k is None


def test_model_param_count():
    config = get_test_config()
    model = Transformer(config)
    num_params = sum(p.numel() for p in model.parameters())
    assert num_params > 0


if __name__ == "__main__":
    pytest.main([__file__])
