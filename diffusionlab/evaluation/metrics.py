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

    is_conditional = hasattr(model, "label_emb") and model.label_emb is not None

    for i, batch in enumerate(data_loader):
        if max_batches is not None and i >= max_batches:
            break
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x_0, y = batch[0], batch[1].to(device)
        else:
            x_0, y = (batch[0] if isinstance(batch, (list, tuple)) else batch), None
        x_0 = x_0.to(device)
        B = x_0.shape[0]

        t = torch.randint(0, forward_diff.timesteps, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(x_0)
        x_t = forward_diff.q_sample(x_0, t, noise=noise)

        pred_noise = model(x_t, t, y=y) if is_conditional else model(x_t, t)

        total_mse += nn.functional.mse_loss(pred_noise, noise, reduction="sum").item()
        total_l1 += nn.functional.l1_loss(pred_noise, noise, reduction="sum").item()
        total_samples += B * x_0.shape[1] * x_0.shape[2] * x_0.shape[3]

    return {
        "mse_loss": total_mse / max(1, total_samples),
        "l1_loss": total_l1 / max(1, total_samples),
    }


def compute_intra_class_diversity(samples: torch.Tensor) -> float:
    """Computes mean pairwise Euclidean distance normalized per pixel across generated samples.

    Higher values indicate greater sample diversity (less mode collapse).
    """
    if samples.shape[0] < 2:
        return 0.0

    flat = samples.flatten(1).float()  # (N, D)
    N, D = flat.shape

    # Pairwise squared distances: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    dot_prod = torch.matmul(flat, flat.t())
    sq_norms = torch.diag(dot_prod)
    dist_sq = sq_norms.unsqueeze(1) - 2 * dot_prod + sq_norms.unsqueeze(0)
    dist = torch.sqrt(torch.clamp(dist_sq, min=0.0))

    # Extract upper triangular entries (excluding diagonal)
    indices = torch.triu_indices(N, N, offset=1)
    mean_dist = dist[indices[0], indices[1]].mean().item()
    normalized_diversity = mean_dist / (D**0.5)
    return round(float(normalized_diversity), 6)


def compute_contrast_and_saturation(
    samples: torch.Tensor,
    saturation_threshold: float = 0.95,
) -> dict[str, float]:
    """Computes spatial contrast and pixel saturation metrics on generated samples [-1, 1].

    Args:
        samples: Tensor of shape (N, C, H, W) normalized in range [-1, 1].
        saturation_threshold: Threshold above which pixels are considered saturated.

    Returns:
        Dict containing:
            - 'mean_contrast': Average spatial standard deviation of pixel intensities.
            - 'saturation_rate': Fraction of pixels where |x| > saturation_threshold.
            - 'dynamic_range': Average peak-to-peak intensity span (max - min).
    """
    flat = samples.flatten(1).float()
    contrast = flat.std(dim=1).mean().item()
    saturation = (flat.abs() > saturation_threshold).float().mean().item()
    dynamic_range = (flat.max(dim=1).values - flat.min(dim=1).values).mean().item()

    return {
        "mean_contrast": round(float(contrast), 6),
        "saturation_rate": round(float(saturation), 6),
        "dynamic_range": round(float(dynamic_range), 6),
    }


@torch.no_grad()
def evaluate_guidance_vector_dynamics(
    model: nn.Module,
    forward_diff: ForwardDiffusion,
    data_loader: DataLoader,
    device: torch.device | str = "cpu",
    timesteps: list[int] | None = None,
    max_batches: int = 5,
) -> dict[int, float]:
    """Evaluates the magnitude of the guidance steering vector ||eps_cond - eps_uncond|| over timesteps.

    Reveals at which diffusion phases classifier-free guidance has the strongest influence.
    """
    device = torch.device(device)
    model.eval()

    if not hasattr(model, "null_label") or model.null_label is None:
        return {}

    null_label = model.null_label
    total_timesteps = forward_diff.timesteps

    if timesteps is None:
        timesteps = [
            int(total_timesteps * ratio)
            for ratio in [0.05, 0.15, 0.25, 0.50, 0.75, 0.90, 0.98]
            if int(total_timesteps * ratio) < total_timesteps
        ]

    dynamics: dict[int, list[float]] = {t_val: [] for t_val in timesteps}

    for i, batch in enumerate(data_loader):
        if i >= max_batches:
            break
        if not (isinstance(batch, (list, tuple)) and len(batch) >= 2):
            continue
        x_0, y = batch[0].to(device), batch[1].to(device)
        B = x_0.shape[0]

        for t_val in timesteps:
            t = torch.full((B,), t_val, device=device, dtype=torch.long)
            noise = torch.randn_like(x_0)
            x_t = forward_diff.q_sample(x_0, t, noise=noise)

            null_y = torch.full_like(y, null_label)
            pred_cond = model(x_t, t, y=y)
            pred_uncond = model(x_t, t, y=null_y)

            # Steering vector norm ||eps_cond - eps_uncond|| normalized per pixel
            diff = (pred_cond - pred_uncond).flatten(1)
            norm = diff.norm(dim=1).mean().item() / (diff.shape[1] ** 0.5)
            dynamics[t_val].append(norm)

    return {t_val: round(sum(vals) / max(1, len(vals)), 6) for t_val, vals in dynamics.items() if vals}

