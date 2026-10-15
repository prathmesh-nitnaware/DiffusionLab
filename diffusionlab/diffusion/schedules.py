"""Noise schedules and diffusion coefficients for DDPM and DDIM."""

import math
from typing import Literal

import torch


def linear_beta_schedule(
    timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Standard linear beta schedule from Ho et al. (2020).

    Args:
        timesteps: Total number of diffusion steps T.
        beta_start: Starting variance beta_1.
        beta_end: Ending variance beta_T.
        dtype: Tensor data type.

    Returns:
        1D Tensor of shape [timesteps] containing beta values in (0, 1).
    """
    if timesteps <= 0:
        raise ValueError(f"timesteps must be positive, got {timesteps}")
    if not (0.0 < beta_start < beta_end < 1.0):
        raise ValueError(
            f"Expected 0 < beta_start < beta_end < 1.0, got beta_start={beta_start}, beta_end={beta_end}"
        )
    return torch.linspace(beta_start, beta_end, timesteps, dtype=dtype)


def cosine_beta_schedule(
    timesteps: int,
    s: float = 0.008,
    max_beta: float = 0.999,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Improved cosine beta schedule from Nichol & Dhariwal (2021).

    Computes alpha_bar_t = f(t) / f(0) where f(t) = cos^2(((t/T + s) / (1 + s)) * (pi / 2)),
    and recovers beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}.

    Args:
        timesteps: Total number of diffusion steps T.
        s: Small offset to prevent beta from being too small near t=0.
        max_beta: Upper clip threshold for beta to prevent singularities at t=T.
        dtype: Tensor data type.

    Returns:
        1D Tensor of shape [timesteps] containing beta values in (0, max_beta].
    """
    if timesteps <= 0:
        raise ValueError(f"timesteps must be positive, got {timesteps}")
    if s < 0.0:
        raise ValueError(f"s offset must be non-negative, got {s}")

    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    f_t = torch.cos(((t / timesteps + s) / (1.0 + s)) * (math.pi / 2.0)) ** 2
    alphas_cumprod = f_t / f_t[0]

    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clip(betas, 1e-5, max_beta)
    return betas.to(dtype=dtype)


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size | tuple[int, ...]) -> torch.Tensor:
    """Extract coefficients from a 1D tensor `a` at batch indices `t` and broadcast to `x_shape`.

    Args:
        a: 1D tensor of shape [T].
        t: 1D tensor of integer timesteps of shape [B].
        x_shape: Shape of target tensor [B, C, H, W] or similar.

    Returns:
        Tensor of shape [B, 1, 1, 1, ...] matching len(x_shape).
    """
    if t.ndim != 1:
        raise ValueError(f"Expected 1D timestep tensor, got shape {t.shape}")
    if (t < 0).any() or (t >= len(a)).any():
        raise IndexError(
            f"Timestep index out of bounds [0, {len(a) - 1}]: min={t.min().item()}, max={t.max().item()}"
        )

    batch_size = t.shape[0]
    out = a.to(device=t.device)[t]
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class BetaSchedule:
    """Precomputed diffusion coefficients for DDPM / DDIM."""

    def __init__(
        self,
        schedule_type: Literal["linear", "cosine"] = "linear",
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        cosine_s: float = 0.008,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.schedule_type = schedule_type
        self.timesteps = timesteps
        self.device = torch.device(device)
        self.dtype = dtype

        if schedule_type == "linear":
            betas = linear_beta_schedule(timesteps, beta_start, beta_end, dtype=torch.float64)
        elif schedule_type == "cosine":
            betas = cosine_beta_schedule(timesteps, s=cosine_s, dtype=torch.float64)
        else:
            raise ValueError(f"Unknown schedule type '{schedule_type}'. Use 'linear' or 'cosine'.")

        self._init_coefficients(betas)

    def _init_coefficients(self, betas: torch.Tensor):
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], dtype=torch.float64), alphas_cumprod[:-1]])

        self.betas = betas.to(dtype=self.dtype, device=self.device)
        self.alphas = alphas.to(dtype=self.dtype, device=self.device)
        self.alphas_cumprod = alphas_cumprod.to(dtype=self.dtype, device=self.device)
        self.alphas_cumprod_prev = alphas_cumprod_prev.to(dtype=self.dtype, device=self.device)

        # Calculations for diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(dtype=self.dtype, device=self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(dtype=self.dtype, device=self.device)
        self.log_one_minus_alphas_cumprod = torch.log(torch.clamp(1.0 - alphas_cumprod, min=1e-20)).to(
            dtype=self.dtype, device=self.device
        )
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod).to(dtype=self.dtype, device=self.device)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1.0).to(
            dtype=self.dtype, device=self.device
        )

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        # posterior_variance = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / torch.clamp(1.0 - alphas_cumprod, min=1e-20)
        self.posterior_variance = posterior_variance.to(dtype=self.dtype, device=self.device)
        # Clipped because posterior variance is 0 at the beginning of the diffusion chain (t=0)
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([posterior_variance[1:2], posterior_variance[1:]])
        ).to(dtype=self.dtype, device=self.device)

        self.posterior_mean_coef1 = (
            betas * torch.sqrt(alphas_cumprod_prev) / torch.clamp(1.0 - alphas_cumprod, min=1e-20)
        ).to(dtype=self.dtype, device=self.device)
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / torch.clamp(1.0 - alphas_cumprod, min=1e-20)
        ).to(dtype=self.dtype, device=self.device)

    def to(self, device: torch.device | str) -> "BetaSchedule":
        self.device = torch.device(device)
        for attr in [
            "betas",
            "alphas",
            "alphas_cumprod",
            "alphas_cumprod_prev",
            "sqrt_alphas_cumprod",
            "sqrt_one_minus_alphas_cumprod",
            "log_one_minus_alphas_cumprod",
            "sqrt_recip_alphas_cumprod",
            "sqrt_recipm1_alphas_cumprod",
            "posterior_variance",
            "posterior_log_variance_clipped",
            "posterior_mean_coef1",
            "posterior_mean_coef2",
        ]:
            setattr(self, attr, getattr(self, attr).to(self.device))
        return self


def get_beta_schedule(
    schedule_type: Literal["linear", "cosine"] = "linear",
    timesteps: int = 1000,
    **kwargs,
) -> BetaSchedule:
    """Helper factory for BetaSchedule."""
    return BetaSchedule(schedule_type=schedule_type, timesteps=timesteps, **kwargs)
