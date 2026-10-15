"""Forward diffusion process implementation."""

from collections.abc import Sequence

import torch

from .schedules import BetaSchedule, extract


class ForwardDiffusion:
    """Implements the forward noising process q(x_t | x_0) and posterior q(x_{t-1} | x_t, x_0)."""

    def __init__(self, schedule: BetaSchedule):
        self.schedule = schedule

    @property
    def timesteps(self) -> int:
        return self.schedule.timesteps

    @property
    def device(self) -> torch.device:
        return self.schedule.device

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample from q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I).

        Equation:
            x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

        Args:
            x_start: Clean input images [B, C, H, W].
            t: Discrete timesteps [B], with 0 <= t < T.
            noise: Optional explicit Gaussian noise [B, C, H, W] for reproducibility.

        Returns:
            Noisy images x_t of shape [B, C, H, W].
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        elif noise.shape != x_start.shape:
            raise ValueError(
                f"Noise shape {noise.shape} does not match x_start shape {x_start.shape}"
            )

        if t.ndim != 1 or t.shape[0] != x_start.shape[0]:
            raise ValueError(
                f"Expected 1D timesteps of length {x_start.shape[0]}, got shape {t.shape}"
            )

        # Make sure t is on same device as schedule
        t = t.to(device=self.device)
        x_start = x_start.to(device=self.device)
        noise = noise.to(device=self.device)

        sqrt_alpha_bar = extract(self.schedule.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_bar = extract(
            self.schedule.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise

    def predict_start_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Recover predicted clean image x_0 from noisy image x_t and predicted noise epsilon.

        Equation:
            x_0 = (x_t - sqrt(1 - alpha_bar_t) * epsilon) / sqrt(alpha_bar_t)
        """
        t = t.to(device=self.device)
        x_t = x_t.to(device=self.device)
        noise = noise.to(device=self.device)

        sqrt_recip_alpha_bar = extract(
            self.schedule.sqrt_recip_alphas_cumprod, t, x_t.shape
        )
        sqrt_recipm1_alpha_bar = extract(
            self.schedule.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )
        return sqrt_recip_alpha_bar * x_t - sqrt_recipm1_alpha_bar * noise

    def q_posterior_mean_variance(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the mean and variance of the true posterior q(x_{t-1} | x_t, x_0)."""
        t = t.to(device=self.device)
        x_start = x_start.to(device=self.device)
        x_t = x_t.to(device=self.device)

        posterior_mean = (
            extract(self.schedule.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.schedule.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.schedule.posterior_variance, t, x_t.shape)
        posterior_log_variance = extract(
            self.schedule.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance

    def sample_trajectory(
        self,
        x_start: torch.Tensor,
        timesteps_list: Sequence[int],
        fixed_noise: torch.Tensor | None = None,
    ) -> list[tuple[int, torch.Tensor]]:
        """Generate a sequence of noisy images at specified timesteps for visualization.

        Args:
            x_start: Single clean image [1, C, H, W] or batch [B, C, H, W].
            timesteps_list: Ascending list of timesteps, e.g. [0, 50, 100, 250, 500, 750, 999].
            fixed_noise: If provided, use the same noise tensor across all steps.

        Returns:
            List of (t, x_t) pairs.
        """
        results = []
        noise = fixed_noise if fixed_noise is not None else torch.randn_like(x_start)
        for t_val in timesteps_list:
            if not (0 <= t_val < self.timesteps):
                raise ValueError(f"t={t_val} out of range [0, {self.timesteps - 1}]")
            t_tensor = torch.full(
                (x_start.shape[0],), t_val, device=self.device, dtype=torch.long
            )
            x_t = self.q_sample(x_start, t_tensor, noise=noise)
            results.append((t_val, x_t))
        return results
