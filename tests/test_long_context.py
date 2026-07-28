import torch

from src.model import LongContextConfig, ModelConfig, RoPEScalingConfig, ThinkingConfig, Transformer


def get_config() -> ModelConfig:
    return ModelConfig(
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=1000,
        max_seq_len=128,
        norm_eps=1e-5,
        use_flash_attn=False,
        dtype="float32",
        long_context=LongContextConfig(
            enabled=True,
            max_seq_len=4096,
            sliding_window_size=64,
            rope_scaling=RoPEScalingConfig(type="yarn", factor=32.0, original_max_seq_len=128),
        ),
        thinking=ThinkingConfig(
            enabled=False,
            thinking_token_budget=512,
            thinking_token_id=999,
            end_think_token_id=998,
            answer_token_id=997,
            end_answer_token_id=996,
        ),
    )


def test_yarn_rope():
    from src.model.rope import precompute_freqs_cis

    freqs = precompute_freqs_cis(
        dim=64,
        max_seq_len=4096,
        theta=10000.0,
        scaling_type="yarn",
        scaling_factor=32.0,
        original_max_seq_len=128,
    )
    assert freqs.shape == (4096, 32)


def test_long_context_config():
    config = get_config()
    assert config.effective_max_seq_len == 4096
    assert config.long_context.sliding_window_size == 64


def test_sliding_window_attention():
    from src.model.attention import Attention

    config = get_config()
    config.long_context.sliding_window_size = 32
    attn = Attention(config, layer_id=0)

    batch_size, seq_len = 2, 64
    x = torch.randn(batch_size, seq_len, config.dim)

    mask = attn._sliding_window_mask(seq_len, 0, x.device, x.dtype)
    if mask is not None:
        assert mask.shape[2] == 64
        assert mask.shape[3] == 64


def test_hierarchical_memory():
    from src.model.context import HierarchicalMemory, compress_hidden

    config = get_config()
    memory = HierarchicalMemory(config)

    hidden = torch.randn(1, 64, 64)
    memory.add_segment(hidden, 0)
    assert len(memory.memories) == 1

    compressed = compress_hidden(hidden, compress_ratio=16)
    assert compressed.shape[1] <= 4


def test_model_with_long_context():
    config = get_config()
    model = Transformer(config)
    model.eval()

    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits, aux_loss = model(input_ids)

    assert logits.shape == (batch_size, seq_len, config.vocab_size)


def test_model_thinking_mode():
    config = get_config()
    config.thinking.enabled = True
    config.thinking.thinking_token_id = 999
    config.thinking.end_think_token_id = 998
    model = Transformer(config)
    model.eval()

    batch_size, seq_len = 1, 8
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        output, thinking = model.generate_with_thinking(
            input_ids, max_new_tokens=16, temperature=1.0, thinking_budget=32
        )
    assert output.shape[0] == batch_size
    assert output.shape[1] > seq_len


def test_deep_think():
    config = get_config()
    config.thinking.deep_think_enabled = True
    config.thinking.deep_think_hypotheses = 2
    model = Transformer(config)
    model.eval()

    batch_size, seq_len = 1, 8
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        output, report = model.deep_think(
            input_ids, max_new_tokens=8, temperature=1.0, n_hypotheses=2
        )
    assert output.shape[0] == batch_size
    assert "Deep Think" in report
