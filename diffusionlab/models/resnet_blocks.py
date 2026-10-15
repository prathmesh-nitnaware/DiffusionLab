"""Residual blocks, attention, downsampling, and upsampling modules for U-Net."""

import math

import torch
import torch.nn.functional as F
from torch import nn


def get_group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """Find the largest divisor of channels up to max_groups for nn.GroupNorm."""
    num_groups = min(max_groups, channels)
    while channels % num_groups != 0 and num_groups > 1:
        num_groups -= 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=channels)


class ResnetBlock(nn.Module):
    """Wide-ResNet style residual block conditioned on timestep embedding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = get_group_norm(in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )

        self.norm2 = get_group_norm(out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """Args:

            x: Input feature maps [B, in_channels, H, W].
            time_emb: Timestep embedding [B, time_emb_dim].

        Returns:
            Output feature maps [B, out_channels, H, W].
        """
        h = self.conv1(self.act1(self.norm1(x)))

        # Project time embedding and add to spatial feature maps
        time_features = self.time_proj(time_emb)[:, :, None, None]
        h = h + time_features

        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.residual_conv(x)


class AttentionBlock(nn.Module):
    """Self-attention block for spatial feature maps."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        if channels % num_heads != 0:
            self.num_heads = math.gcd(channels, num_heads)

        self.norm = get_group_norm(channels)
        self.qkv_conv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x

        h = self.norm(x)
        qkv = self.qkv_conv(h)  # [B, 3*C, H, W]
        q, k, v = torch.chunk(qkv, 3, dim=1)

        # Reshape to [B, num_heads, H*W, head_dim]
        head_dim = C // self.num_heads
        q = q.view(B, self.num_heads, head_dim, H * W).transpose(2, 3)
        k = k.view(B, self.num_heads, head_dim, H * W).transpose(2, 3)
        v = v.view(B, self.num_heads, head_dim, H * W).transpose(2, 3)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, v)  # [B, num_heads, H*W, head_dim]

        out = out.transpose(2, 3).contiguous().view(B, C, H, W)
        out = self.proj_out(out)
        return out + residual


class Downsample(nn.Module):
    """Spatial downsampling by a factor of 2."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Spatial upsampling by a factor of 2."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)
