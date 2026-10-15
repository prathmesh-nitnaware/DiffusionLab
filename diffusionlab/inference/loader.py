"""Checkpoint model restoration and loader utilities."""

from pathlib import Path
from typing import Any

import torch

from diffusionlab.diffusion.schedules import BetaSchedule, get_beta_schedule
from diffusionlab.models.unet import UNet
from diffusionlab.training.checkpointing import load_checkpoint


def is_diffusion_checkpoint(checkpoint_path: str | Path) -> bool:
    """Validate whether a file is a genuine diffusion U-Net checkpoint."""
    path = Path(checkpoint_path)
    if not path.is_file() or not path.name.endswith(".pt"):
        return False
    # Explicitly reject classifier checkpoints
    if "classifier" in path.parts or "classifier" in path.name.lower():
        return False
    try:
        raw = torch.load(path, map_location="cpu")
        if not isinstance(raw, dict):
            return False
        state_dict = raw.get("model_state_dict", raw)
        return "init_conv.weight" in state_dict and any(k.startswith("time_mlp") for k in state_dict)
    except Exception:
        return False


def get_checkpoint_info(checkpoint_path: str | Path) -> dict[str, Any]:
    """Extract metadata and architecture configuration from a diffusion checkpoint."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    raw = torch.load(path, map_location="cpu")
    config = raw.get("config", {})
    model_cfg = config.get("model", {})
    diff_cfg = config.get("diffusion", {})
    state_dict = raw.get("model_state_dict", {})

    is_valid = "init_conv.weight" in state_dict and any(k.startswith("time_mlp") for k in state_dict)
    if not is_valid:
        return {
            "path": str(path),
            "name": path.name,
            "compatible": False,
            "error": "Not a diffusion U-Net checkpoint",
        }

    is_conditional = "label_emb.weight" in state_dict or model_cfg.get("num_classes") is not None
    num_classes = model_cfg.get("num_classes")
    if num_classes is None and "label_emb.weight" in state_dict:
        num_classes = state_dict["label_emb.weight"].shape[0] - 1

    return {
        "path": str(path),
        "name": path.name,
        "run_name": path.parent.parent.name if path.parent.name == "checkpoints" else path.parent.name,
        "compatible": True,
        "epoch": raw.get("epoch", "N/A"),
        "global_step": raw.get("global_step", "N/A"),
        "loss": round(float(raw.get("loss")), 4) if raw.get("loss") is not None else "N/A",
        "is_conditional": is_conditional,
        "num_classes": num_classes,
        "base_channels": state_dict["init_conv.weight"].shape[0],
        "timesteps": diff_cfg.get("timesteps", 200),
        "schedule_type": diff_cfg.get("schedule_type", "cosine"),
        "config": config,
    }


def find_diffusion_checkpoints(search_dirs: list[str | Path] | None = None) -> list[dict[str, Any]]:
    """Discover all valid diffusion U-Net checkpoints, strictly excluding classifiers and incompatible weights."""
    if search_dirs is None:
        search_dirs = ["runs", "checkpoints"]

    discovered: list[dict[str, Any]] = []
    seen_paths = set()

    for d in search_dirs:
        dir_path = Path(d)
        if not dir_path.exists():
            continue
        for pt_path in dir_path.rglob("*.pt"):
            if "classifier" in pt_path.parts or "classifier" in pt_path.name.lower():
                continue
            canonical = str(pt_path.resolve())
            if canonical in seen_paths:
                continue
            seen_paths.add(canonical)

            if is_diffusion_checkpoint(pt_path):
                info = get_checkpoint_info(pt_path)
                discovered.append(info)

    # Sort preferred checkpoints first: mini-project best, conditional best, baseline best
    def sort_priority(item: dict[str, Any]) -> tuple[int, str]:
        p = item["path"].lower()
        if "mini_project" in p and "best" in p:
            return (0, p)
        if "conditional" in p and "best" in p:
            return (1, p)
        if "baseline" in p and "best" in p:
            return (2, p)
        if "best" in p:
            return (3, p)
        return (4, p)

    return sorted(discovered, key=sort_priority)


def load_model_and_schedule_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[UNet, BetaSchedule, dict[str, Any]]:
    """Inspects checkpoint, reconstructs compatible UNet and BetaSchedule, and loads weights."""
    device = torch.device(device)
    raw_ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = raw_ckpt.get("model_state_dict", {})

    # Validate diffusion architecture compatibility
    if "init_conv.weight" not in state_dict or not any(k.startswith("time_mlp") for k in state_dict):
        raise ValueError(
            f"Checkpoint at '{checkpoint_path}' is incompatible with DiffusionLab U-Net. "
            f"Found state_dict keys: {list(state_dict.keys())[:5]}... "
            f"It appears to be a classifier or foreign model."
        )

    config = raw_ckpt.get("config", {})
    model_cfg = config.get("model", {})
    diff_cfg = config.get("diffusion", {})

    # Detect base channels and input channels from init_conv
    base_channels = state_dict["init_conv.weight"].shape[0]
    in_channels = state_dict["init_conv.weight"].shape[1]
    out_channels = model_cfg.get("out_channels", in_channels)

    channel_multipliers = model_cfg.get("channel_multipliers", (1, 2))
    num_res_blocks = model_cfg.get("num_res_blocks", 1)
    attention_resolutions = model_cfg.get("attention_resolutions", ())

    num_classes = model_cfg.get("num_classes", None)
    if num_classes is None and "label_emb.weight" in state_dict:
        num_classes = state_dict["label_emb.weight"].shape[0] - 1

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
        num_classes=num_classes,
        dropout=0.0,
    ).to(device)

    load_checkpoint(checkpoint_path, model=model, device=device)
    model.eval()
    return model, schedule, raw_ckpt
