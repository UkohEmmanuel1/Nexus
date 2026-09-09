import torch


def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device = None,
    scaling_type: str = None,
    scaling_factor: float = 1.0,
    original_max_seq_len: int = 4096,
    beta_fast: int = 32,
    beta_slow: int = 1,
) -> torch.Tensor:
    if scaling_type in ("yarn", "ntk"):
        return _precompute_freqs_cis_scaled(
            dim,
            max_seq_len,
            theta,
            device,
            scaling_type,
            scaling_factor,
            original_max_seq_len,
            beta_fast,
            beta_slow,
        )

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def _precompute_freqs_cis_scaled(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device = None,
    scaling_type: str = "yarn",
    scaling_factor: float = 32.0,
    original_max_seq_len: int = 4096,
    beta_fast: int = 32,
    beta_slow: int = 1,
) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))

    if scaling_type == "ntk":
        theta_scale = scaling_factor ** (dim / (dim - 2))
        freqs = 1.0 / (
            (theta * theta_scale) ** (torch.arange(0, dim, 2, device=device).float() / dim)
        )
    elif scaling_type == "yarn":
        t = torch.arange(original_max_seq_len, device=device, dtype=torch.float32)
        t = t / scaling_factor

        _freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
        freqs_out = torch.outer(t, _freqs)
        torch.polar(torch.ones_like(freqs_out), freqs_out)

        dims = dim // 2
        wave_len = 2 * torch.pi * theta ** (2 * torch.arange(dims, device=device).float() / dim)
        alpha = wave_len / original_max_seq_len

        ramp = torch.sigmoid(beta_fast * (1 - alpha)) * torch.sigmoid(beta_slow * (alpha - 1))
        ramp = 1 - ramp
        smooth_interpolation = scaling_factor - (scaling_factor - 1) * ramp

        t_ext = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        t_ext_scaled = t_ext.unsqueeze(-1) / smooth_interpolation.unsqueeze(0)
        freqs_ext = t_ext_scaled * _freqs.unsqueeze(0)
        freqs_cis = torch.polar(torch.ones_like(freqs_ext), freqs_ext)
        return freqs_cis

    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)
