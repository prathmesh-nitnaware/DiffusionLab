"""Checkpoint model restoration and loader utilities."""

from pathlib import Path
from typing import Any

import torch

from diffusionlab.diffusion.schedules import BetaSchedule, get_beta_schedule
from diffusionlab.models.unet import UNet
from diffusionlab.training.checkpointing import load_checkpoint


def load_model_and_schedule_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[UNet, BetaSchedule, dict[str, Any]]:
    """Inspects checkpoint, reconstructs compatible UNet and BetaSchedule, and loads weights."""
    device = torch.device(device)
    raw_ckpt = torch.load(checkpoint_path, map_location=device)
    config = raw_ckpt.get("config", {})
    model_cfg = config.get("model", {})
    diff_cfg = config.get("diffusion", {})

    state_dict = raw_ckpt.get("model_state_dict", {})

    # Detect base channels and input channels from init_conv if not in config
    if "init_conv.weight" in state_dict:
        base_channels = state_dict["init_conv.weight"].shape[0]
        in_channels = state_dict["init_conv.weight"].shape[1]
    else:
        base_channels = model_cfg.get("base_channels", 32)
        in_channels = model_cfg.get("in_channels", 1)

    out_channels = model_cfg.get("out_channels", in_channels)

    # Detect channel multipliers by looking at downs
    # For instance downs.0.res.conv1.weight etc.
    if "channel_multipliers" in model_cfg:
        channel_multipliers = model_cfg["channel_multipliers"]
    else:
        # Determine from downs if available or default (1, 2)
        channel_multipliers = (1, 2)

    num_res_blocks = model_cfg.get("num_res_blocks", 1)
    attention_resolutions = model_cfg.get("attention_resolutions", ())

    schedule_type = diff_cfg.get("schedule_type", "cosine")
    timesteps = diff_cfg.get("timesteps", 200)

    schedule = get_beta_schedule(
        schedule_type=schedule_type,
        timesteps=timesteps,
        beta_start=diff_cfg.get("beta_start", 1e-4),
        beta_end=diff_cfg.get("beta_end", 0.02),
        cosine_s=diff_cfg.get("cosine_s", 0.008),
        device=device,
    )

    model = UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        channel_multipliers=channel_multipliers,
        num_res_blocks=num_res_blocks,
        attention_resolutions=attention_resolutions,
        dropout=0.0,
    ).to(device)

    load_checkpoint(checkpoint_path, model=model, device=device)
    model.eval()
    return model, schedule, raw_ckpt
