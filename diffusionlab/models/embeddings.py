"""Timestep embeddings for diffusion models."""

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for scalar timesteps, as in Vaswani et al. / Ho et al. (2020).

    Maps integer/scalar timestep t to embedding of dimension dim:
        PE(t, 2i) = sin(t / 10000^(2i/dim))
        PE(t, 2i+1) = cos(t / 10000^(2i/dim))
    """

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Embedding dimension must be even, got {dim}")
        self.dim = dim
        self.max_period = max_period

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Args:

            timesteps: 1D tensor of shape [B] containing timesteps.

        Returns:
            Tensor of shape [B, dim].
        """
        if timesteps.ndim != 1:
            timesteps = timesteps.view(-1)

        half_dim = self.dim // 2
        # Compute frequencies in log space for numerical stability:
        # freqs = exp(-log(max_period) * i / half_dim)
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
            / half_dim
        )
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding


class TimestepMLP(nn.Module):
    """Trainable 2-layer MLP projection for sinusoidal timestep embeddings."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.mlp = nn.Sequential(
            SinusoidalTimeEmbedding(in_features),
            nn.Linear(in_features, out_features),
            nn.SiLU(),
            nn.Linear(out_features, out_features),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.mlp(timesteps)
