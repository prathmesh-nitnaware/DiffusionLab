"""Evaluation metrics and model profiling for diffusion models."""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from diffusionlab.diffusion.forward_process import ForwardDiffusion


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable_parameters": trainable, "total_parameters": total}


def get_checkpoint_size(checkpoint_path: str | Path) -> dict[str, float]:
    """Get checkpoint file size in bytes and megabytes."""
    path = Path(checkpoint_path)
    if not path.is_file():
        return {"size_bytes": 0, "size_mb": 0.0}
    size_bytes = path.stat().st_size
    return {"size_bytes": size_bytes, "size_mb": size_bytes / (1024 * 1024)}


def get_memory_stats(device: torch.device | str) -> dict[str, float]:
    """Get GPU memory statistics in megabytes (returns 0 for CPU)."""
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)
        reserved = torch.cuda.memory_reserved(device) / (1024 * 1024)
        max_allocated = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        return {
            "allocated_mb": round(allocated, 2),
            "reserved_mb": round(reserved, 2),
            "max_allocated_mb": round(max_allocated, 2),
        }
    return {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0}


@torch.no_grad()
def evaluate_noise_prediction(
    model: nn.Module,
    forward_diff: ForwardDiffusion,
    data_loader: DataLoader,
    device: torch.device | str = "cpu",
    max_batches: int | None = None,
) -> dict[str, float]:
    """Evaluate noise prediction MSE and L1 errors over a dataset split.

    Returns:
        Dict with 'mse_loss' and 'l1_loss'.
    """
    device = torch.device(device)
    model.eval()
    total_mse = 0.0
    total_l1 = 0.0
    total_samples = 0

    for i, batch in enumerate(data_loader):
        if max_batches is not None and i >= max_batches:
            break
        x_0 = batch[0] if isinstance(batch, (list, tuple)) else batch
        x_0 = x_0.to(device)
        B = x_0.shape[0]

        t = torch.randint(0, forward_diff.timesteps, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(x_0)
        x_t = forward_diff.q_sample(x_0, t, noise=noise)

        pred_noise = model(x_t, t)

        total_mse += nn.functional.mse_loss(pred_noise, noise, reduction="sum").item()
        total_l1 += nn.functional.l1_loss(pred_noise, noise, reduction="sum").item()
        total_samples += B * x_0.shape[1] * x_0.shape[2] * x_0.shape[3]

    return {
        "mse_loss": total_mse / max(1, total_samples),
        "l1_loss": total_l1 / max(1, total_samples),
    }
