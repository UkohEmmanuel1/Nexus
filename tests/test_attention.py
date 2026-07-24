import pytest
import torch

from src.model import Attention, ModelConfig
from src.model.rope import precompute_freqs_cis


def get_config() -> ModelConfig:
    return ModelConfig(
        dim=256,
        n_heads=8,
        n_kv_heads=4,
        max_seq_len=64,
        use_flash_attn=False,
        dtype="float32",
    )


def test_attention_shape():
    config = get_config()
    attn = Attention(config, layer_id=0)

    batch_size, seq_len = 2, 16
    x = torch.randn(batch_size, seq_len, config.dim)
    freqs_cis = precompute_freqs_cis(config.head_dim, config.max_seq_len)

    output = attn(x, freqs_cis)
    assert output.shape == (batch_size, seq_len, config.dim)


def test_attention_with_mask():
    config = get_config()
    attn = Attention(config, layer_id=0)

    batch_size, seq_len = 2, 16
    x = torch.randn(batch_size, seq_len, config.dim)
    freqs_cis = precompute_freqs_cis(config.head_dim, config.max_seq_len)
    mask = torch.full((seq_len, seq_len), float("-inf"))
    mask = torch.triu(mask, diagonal=1)
    mask = mask.unsqueeze(0).unsqueeze(0)

    output = attn(x, freqs_cis, mask=mask)
    assert output.shape == (batch_size, seq_len, config.dim)


def test_attention_kv_cache():
    config = get_config()
    attn = Attention(config, layer_id=0)

    batch_size = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    attn.to(device)
    attn.init_kv_cache(batch_size, config.max_seq_len, device, torch.float32)

    seq_len_1, seq_len_2 = 8, 4
    x1 = torch.randn(batch_size, seq_len_1, config.dim, device=device)
    x2 = torch.randn(batch_size, seq_len_2, config.dim, device=device)

    freqs_cis = precompute_freqs_cis(config.head_dim, config.max_seq_len, device=device)

    out1 = attn(x1, freqs_cis, start_pos=0, use_cache=True)
    out2 = attn(x2, freqs_cis, start_pos=seq_len_1, use_cache=True)

    assert out1.shape == (batch_size, seq_len_1, config.dim)
    assert out2.shape == (batch_size, seq_len_2, config.dim)

    attn.reset_kv_cache()
    assert attn.cache_k is None


def test_grouped_query_attention():
    config = get_config()
    assert config.n_heads // config.n_kv_heads == 2
    attn = Attention(config, layer_id=0)

    batch_size, seq_len = 2, 8
    x = torch.randn(batch_size, seq_len, config.dim)
    freqs_cis = precompute_freqs_cis(config.head_dim, config.max_seq_len)

    output = attn(x, freqs_cis)
    assert output.shape == (batch_size, seq_len, config.dim)


if __name__ == "__main__":
    pytest.main([__file__])
