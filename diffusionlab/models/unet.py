"""Custom timestep-conditioned U-Net for noise prediction."""

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .embeddings import TimestepMLP
from .resnet_blocks import AttentionBlock, Downsample, ResnetBlock, Upsample, get_group_norm


class UNet(nn.Module):
    """Timestep-conditioned U-Net architecture for diffusion models.

    Takes an input image x_t and timestep t, and predicts the added noise epsilon.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int | None = None,
        base_channels: int = 32,
        channel_multipliers: Sequence[int] = (1, 2, 4),
        num_res_blocks: int = 2,
        attention_resolutions: Sequence[int] = (14, 7),
        dropout: float = 0.0,
        time_emb_dim: int | None = None,
        num_classes: int | None = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.base_channels = base_channels
        self.channel_multipliers = list(channel_multipliers)
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = list(attention_resolutions)
        self.num_classes = num_classes

        if time_emb_dim is None:
            time_emb_dim = base_channels * 4
        self.time_emb_dim = time_emb_dim

        # Timestep embedding MLP
        self.time_mlp = TimestepMLP(base_channels, time_emb_dim)

        # Class label conditioning for Classifier-Free Guidance (CFG)
        if num_classes is not None:
            self.null_label = num_classes  # Token index representing null condition
            self.label_emb = nn.Embedding(num_classes + 1, time_emb_dim)
        else:
            self.null_label = None
            self.label_emb = None

        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Encoder (Downsampling path)
        self.downs = nn.ModuleList()
        current_channels = base_channels
        channels_list = [current_channels]

        for stage_idx, mult in enumerate(self.channel_multipliers):
            stage_channels = base_channels * mult
            for _ in range(num_res_blocks):
                block = nn.ModuleDict({
                    "res": ResnetBlock(
                        current_channels,
                        stage_channels,
                        time_emb_dim=time_emb_dim,
                        dropout=dropout,
                    ),
                    "attn": AttentionBlock(stage_channels) if mult in self.attention_resolutions else nn.Identity(),
                })
                self.downs.append(block)
                current_channels = stage_channels
                channels_list.append(current_channels)

            # Downsample between stages except for the last stage
            if stage_idx < len(self.channel_multipliers) - 1:
                self.downs.append(nn.ModuleDict({"downsample": Downsample(current_channels)}))
                channels_list.append(current_channels)

        # Bottleneck (Middle path)
        self.mid_block1 = ResnetBlock(
            current_channels, current_channels, time_emb_dim=time_emb_dim, dropout=dropout
        )
        self.mid_attn = AttentionBlock(current_channels)
        self.mid_block2 = ResnetBlock(
            current_channels, current_channels, time_emb_dim=time_emb_dim, dropout=dropout
        )

        # Decoder (Upsampling path)
        self.ups = nn.ModuleList()
        for stage_idx, mult in reversed(list(enumerate(self.channel_multipliers))):
            stage_channels = base_channels * mult
            for _ in range(num_res_blocks + 1):
                skip_channels = channels_list.pop()
                block = nn.ModuleDict({
                    "res": ResnetBlock(
                        current_channels + skip_channels,
                        stage_channels,
                        time_emb_dim=time_emb_dim,
                        dropout=dropout,
                    ),
                    "attn": AttentionBlock(stage_channels) if mult in self.attention_resolutions else nn.Identity(),
                })
                self.ups.append(block)
                current_channels = stage_channels

            # Upsample between stages except after the final stage
            if stage_idx > 0:
                self.ups.append(nn.ModuleDict({"upsample": Upsample(current_channels)}))

        # Final projection
        self.final_norm = get_group_norm(current_channels)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(current_channels, self.out_channels, kernel_size=3, padding=1)

    def num_parameters(self, only_trainable: bool = True) -> int:
        """Count parameters in the model."""
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Args:

            x: Input noisy image batch [B, in_channels, H, W].
            timesteps: Timestep tensor [B] (integer or float).
            y: Optional class label tensor [B] for conditional generation.

        Returns:
            Predicted noise [B, out_channels, H, W].
        """
        # Timestep embedding
        cond_emb = self.time_mlp(timesteps)

        # Class condition embedding if model is class-conditioned
        if self.label_emb is not None:
            if y is None:
                # Default to null condition token
                y = torch.full((x.shape[0],), self.null_label, dtype=torch.long, device=x.device)
            else:
                if (y < 0).any() or (y > self.null_label).any():
                    raise ValueError(
                        f"Class labels must be in range [0, {self.null_label}], "
                        f"got min={y.min().item()}, max={y.max().item()}"
                    )
            cond_emb = cond_emb + self.label_emb(y)

        # Initial convolution
        h = self.init_conv(x)
        skips = [h]

        # Downsampling path
        for module in self.downs:
            if "downsample" in module:
                h = module["downsample"](h)
                skips.append(h)
            else:
                h = module["res"](h, cond_emb)
                h = module["attn"](h)
                skips.append(h)

        # Middle bottleneck
        h = self.mid_block1(h, cond_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, cond_emb)

        # Upsampling path
        for module in self.ups:
            if "upsample" in module:
                h = module["upsample"](h)
            else:
                skip = skips.pop()
                # Handle potential 1-pixel discrepancy due to odd spatial resolutions
                if h.shape[2:] != skip.shape[2:]:
                    h = F.interpolate(h, size=skip.shape[2:], mode="nearest")
                h = torch.cat([h, skip], dim=1)
                h = module["res"](h, cond_emb)
                h = module["attn"](h)

        # Final projection
        h = self.final_act(self.final_norm(h))
        return self.final_conv(h)

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        y: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """Predict noise using Classifier-Free Guidance (CFG):

            epsilon_guided = epsilon_uncond + guidance_scale * (epsilon_cond - epsilon_uncond)

        Args:
            x: Noisy image batch [B, C, H, W].
            timesteps: Timestep tensor [B].
            y: Class labels [B]. If None, runs unconditional forward pass.
            guidance_scale: Strength of guidance (1.0 = standard conditional, >1.0 = amplified guidance).

        Returns:
            Guided predicted noise tensor [B, C, H, W].
        """
        if guidance_scale == 1.0 or self.label_emb is None or y is None:
            return self.forward(x, timesteps, y)

        batch_size = x.shape[0]
        # Duplicate batch: [cond_batch, uncond_batch]
        x_in = torch.cat([x, x], dim=0)
        t_in = torch.cat([timesteps, timesteps], dim=0)
        null_y = torch.full_like(y, self.null_label)
        y_in = torch.cat([y, null_y], dim=0)

        # Single batched forward pass for both conditional and unconditional noise
        pred = self.forward(x_in, t_in, y_in)
        pred_cond, pred_uncond = pred[:batch_size], pred[batch_size:]

        # Guidance formula: epsilon_guided = epsilon_uncond + guidance_scale * (epsilon_cond - epsilon_uncond)
        return pred_uncond + guidance_scale * (pred_cond - pred_uncond)

