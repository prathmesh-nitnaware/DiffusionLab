"""Reverse diffusion sampling: DDPM (stochastic) and DDIM (deterministic/accelerated)."""


import torch
from torch import nn
from tqdm import tqdm

from .forward_process import extract
from .schedules import BetaSchedule


class DDPMSampler:
    """Standard Denoising Diffusion Probabilistic Model (DDPM) reverse-diffusion sampler.

    Implements the ancestral sampling algorithm from Ho et al. (2020).
    """

    def __init__(self, model: nn.Module, schedule: BetaSchedule):
        self.model = model
        self.schedule = schedule

    @torch.no_grad()
    def p_sample(
        self,
        x_t: torch.Tensor,
        t_index: int,
        y: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """Sample x_{t-1} from p_theta(x_{t-1} | x_t) at discrete timestep t_index."""
        device = x_t.device
        t_tensor = torch.full((x_t.shape[0],), t_index, device=device, dtype=torch.long)

        # 1. Predict noise epsilon_theta(x_t, t) with optional classifier-free guidance
        if hasattr(self.model, "forward_with_cfg") and y is not None and guidance_scale != 1.0:
            pred_noise = self.model.forward_with_cfg(x_t, t_tensor, y=y, guidance_scale=guidance_scale)
        elif y is not None:
            pred_noise = self.model(x_t, t_tensor, y=y)
        else:
            pred_noise = self.model(x_t, t_tensor)

        # 2. Predict clean image x_0
        sqrt_recip_alpha_bar = extract(self.schedule.sqrt_recip_alphas_cumprod, t_tensor, x_t.shape)
        sqrt_recipm1_alpha_bar = extract(self.schedule.sqrt_recipm1_alphas_cumprod, t_tensor, x_t.shape)
        pred_x0 = sqrt_recip_alpha_bar * x_t - sqrt_recipm1_alpha_bar * pred_noise

        if clip_denoised:
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

        # 3. Compute posterior mean mu_tilde(pred_x0, x_t)
        posterior_mean = (
            extract(self.schedule.posterior_mean_coef1, t_tensor, x_t.shape) * pred_x0
            + extract(self.schedule.posterior_mean_coef2, t_tensor, x_t.shape) * x_t
        )

        # 4. Compute posterior variance and add stochastic noise (except at t=0)
        if t_index == 0:
            return posterior_mean
        else:
            posterior_log_variance = extract(
                self.schedule.posterior_log_variance_clipped, t_tensor, x_t.shape
            )
            noise = torch.randn_like(x_t)
            return posterior_mean + torch.exp(0.5 * posterior_log_variance) * noise

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, ...],
        device: torch.device | str = "cpu",
        y: torch.Tensor | int | None = None,
        guidance_scale: float = 1.0,
        seed: int | None = None,
        clip_denoised: bool = True,
        save_intermediate_steps: int | None = None,
        show_progress: bool = True,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Generate samples from Gaussian noise using ancestral reverse diffusion.

        Args:
            shape: Target sample shape [B, C, H, W].
            device: Computation device.
            y: Optional class labels [B] or int for conditional generation.
            guidance_scale: CFG guidance scale (>1.0 to amplify condition).
            seed: Optional random seed for reproducible generation.
            clip_denoised: Whether to clamp predicted x_0 to [-1, 1].
            save_intermediate_steps: If provided, records intermediate images every N steps.
            show_progress: Whether to display a tqdm progress bar.

        Returns:
            Tuple of (final_samples, list_of_saved_intermediate_tensors).
        """
        if seed is not None:
            torch.manual_seed(seed)

        device = torch.device(device)
        self.model.eval()

        if isinstance(y, int):
            y = torch.full((shape[0],), y, device=device, dtype=torch.long)
        elif isinstance(y, torch.Tensor):
            y = y.to(device)

        # Initialize from standard Gaussian noise x_T ~ N(0, I)
        x = torch.randn(shape, device=device)
        intermediates: list[torch.Tensor] = [x.clone()]

        timesteps = list(reversed(range(self.schedule.timesteps)))
        iterator = tqdm(timesteps, desc="DDPM Sampling", leave=False) if show_progress else timesteps

        for t in iterator:
            x = self.p_sample(
                x,
                t,
                y=y,
                guidance_scale=guidance_scale,
                clip_denoised=clip_denoised,
            )

            if save_intermediate_steps is not None and (
                t % save_intermediate_steps == 0 or t == 0
            ):
                intermediates.append(x.clone())

        return x, intermediates


class DDIMSampler:
    """Denoising Diffusion Implicit Models (DDIM) sampler from Song et al. (2020).

    Enables non-Markovian deterministic or stochastic sampling with accelerated step counts.
    """

    def __init__(self, model: nn.Module, schedule: BetaSchedule):
        self.model = model
        self.schedule = schedule

    def _get_timestep_sequence(self, num_inference_steps: int) -> list[int]:
        """Construct descending sequence of integer timesteps for sampling."""
        total_timesteps = self.schedule.timesteps
        if num_inference_steps > total_timesteps:
            raise ValueError(
                f"num_inference_steps ({num_inference_steps}) cannot exceed total timesteps ({total_timesteps})"
            )
        # Uniformly spaced steps across [0, total_timesteps - 1]
        c = total_timesteps // num_inference_steps
        timesteps = [i * c for i in range(num_inference_steps)]
        # Ensure the last step is T-1
        timesteps[-1] = total_timesteps - 1
        return sorted(set(timesteps), reverse=True)

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, ...],
        num_inference_steps: int = 50,
        eta: float = 0.0,
        y: torch.Tensor | int | None = None,
        guidance_scale: float = 1.0,
        device: torch.device | str = "cpu",
        seed: int | None = None,
        clip_denoised: bool = True,
        save_intermediate: bool = False,
        show_progress: bool = True,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Generate samples using DDIM with optional Classifier-Free Guidance.

        Args:
            shape: Target sample shape [B, C, H, W].
            num_inference_steps: Number of sampling steps (e.g., 50, 100, 250, 1000).
            eta: Weight of stochastic noise. eta=0.0 corresponds to deterministic DDIM (ODE).
            y: Optional class labels [B] or int for conditional generation.
            guidance_scale: Strength of CFG (1.0 = standard, >1.0 = amplified).
            device: Target device.
            seed: Optional seed for reproducible generation.
            clip_denoised: Clamp predicted clean image to [-1, 1].
            save_intermediate: Whether to save intermediate states.
            show_progress: Show tqdm progress bar.

        Returns:
            Tuple of (final_samples, list_of_saved_intermediate_tensors).
        """
        if seed is not None:
            torch.manual_seed(seed)

        device = torch.device(device)
        self.model.eval()

        if isinstance(y, int):
            y = torch.full((shape[0],), y, device=device, dtype=torch.long)
        elif isinstance(y, torch.Tensor):
            y = y.to(device)

        time_seq = self._get_timestep_sequence(num_inference_steps)
        # Previous timesteps: [t_{s-1}, ..., 0, -1]
        time_seq_prev = time_seq[1:] + [-1]

        x = torch.randn(shape, device=device)
        intermediates: list[torch.Tensor] = [x.clone()]

        iterator = (
            tqdm(zip(time_seq, time_seq_prev), total=len(time_seq), desc="DDIM Sampling", leave=False)
            if show_progress
            else zip(time_seq, time_seq_prev)
        )

        alphas_cumprod = self.schedule.alphas_cumprod.to(device)

        for t_curr, t_prev in iterator:
            t_tensor = torch.full((shape[0],), t_curr, device=device, dtype=torch.long)

            # 1. Predict noise epsilon_theta(x_t, t) with optional CFG
            if hasattr(self.model, "forward_with_cfg") and y is not None and guidance_scale != 1.0:
                pred_noise = self.model.forward_with_cfg(x, t_tensor, y=y, guidance_scale=guidance_scale)
            elif y is not None:
                pred_noise = self.model(x, t_tensor, y=y)
            else:
                pred_noise = self.model(x, t_tensor)

            # 2. Extract alpha_bar_t and alpha_bar_prev
            alpha_bar_t = alphas_cumprod[t_curr]
            alpha_bar_prev = alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)

            # 3. Estimate clean image x_0
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

            pred_x0 = (x - sqrt_one_minus_alpha_bar_t * pred_noise) / sqrt_alpha_bar_t
            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

            # 4. Compute DDIM variance sigma_t
            if eta > 0.0 and t_prev >= 0:
                sigma_t = eta * torch.sqrt(
                    ((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)) * (1.0 - alpha_bar_t / alpha_bar_prev)
                )
            else:
                sigma_t = torch.tensor(0.0, device=device)

            # 5. Direction pointing to x_t
            dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma_t**2, min=0.0)) * pred_noise

            # 6. Random noise term
            if sigma_t > 0.0 and t_prev >= 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            # 7. Update x to x_{t_prev}
            x = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt + sigma_t * noise

            if save_intermediate:
                intermediates.append(x.clone())

        return x, intermediates
