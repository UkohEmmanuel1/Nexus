from dataclasses import dataclass, field


@dataclass
class MoEConfig:
    enabled: bool = False
    num_experts: int = 8
    top_k: int = 2
    capacity_factor: float = 1.25
    aux_loss_coef: float = 0.01
    z_loss_coef: float = 0.001


@dataclass
class RoPEScalingConfig:
    type: str = "yarn"
    factor: float = 32.0
    original_max_seq_len: int = 4096
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float | None = None
    mscale_all_dim: float | None = None


@dataclass
class LongContextConfig:
    enabled: bool = True
    max_seq_len: int = 1048576
    rope_scaling: RoPEScalingConfig = field(default_factory=RoPEScalingConfig)
    sliding_window_size: int = 4096
    global_attention_every_n: int = 128
    num_global_heads: int = 4
    compress_ratio: int = 64
    memory_tokens: int = 64
    chunk_size: int = 4096


@dataclass
class ThinkingConfig:
    enabled: bool = True
    thinking_token_budget: int = 2048
    thinking_token_id: int = 128000
    end_think_token_id: int = 128001
    answer_token_id: int = 128002
    end_answer_token_id: int = 128003
    deep_think_enabled: bool = True
    deep_think_hypotheses: int = 3


@dataclass
class ModelConfig:
    dim: int = 3200
    n_layers: int = 26
    n_heads: int = 32
    n_kv_heads: int = 8
    ffn_mult: float = 2.6667
    vocab_size: int = 128000
    max_seq_len: int = 4096
    norm_eps: float = 1.0e-5
    rope_theta: float = 10000.0
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    moe: MoEConfig = field(default_factory=MoEConfig)
    init_std: float = 0.02
    use_flash_attn: bool = True
    gradient_checkpointing: bool = False
    dtype: str = "bfloat16"
    long_context: LongContextConfig = field(default_factory=LongContextConfig)
    thinking: ThinkingConfig = field(default_factory=ThinkingConfig)

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def n_rep(self) -> int:
        return self.n_heads // self.n_kv_heads

    @property
    def ffn_dim(self) -> int:
        return int(self.ffn_mult * self.dim)

    def num_params(self) -> int:
        vocab = self.vocab_size
        d = self.dim
        ffn = self.ffn_dim
        head_dim = self.head_dim
        n_kv = self.n_kv_heads

        embed = vocab * d
        total = embed

        for _ in range(self.n_layers):
            attn = (
                (d * self.n_heads * head_dim) + (d * n_kv * head_dim * 2) + (d * d * self.n_heads)
            )
            ffn_layer = (
                (d * ffn * 3)
                if not self.moe.enabled
                else (d * ffn * 3 * self.moe.num_experts + d * self.moe.num_experts)
            )
            norms = 4 * d
            total += attn + ffn_layer + norms

        total += d * vocab
        return total

    @property
    def effective_max_seq_len(self) -> int:
        if self.long_context.enabled:
            return self.long_context.max_seq_len
        return self.max_seq_len
