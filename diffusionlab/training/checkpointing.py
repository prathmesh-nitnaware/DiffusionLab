"""Checkpoint management for saving and resuming diffusion models."""

from pathlib import Path
from typing import Any

import torch
from torch import nn, optim


def save_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    epoch: int = 0,
    global_step: int = 0,
    loss: float = 0.0,
    config: dict[str, Any] | None = None,
) -> Path:
    """Save full training checkpoint dictionary.

    Args:
        checkpoint_path: Destination file path.
        model: PyTorch model.
        optimizer: PyTorch optimizer.
        scheduler: Learning rate scheduler.
        scaler: GradScaler for mixed precision.
        epoch: Current epoch index.
        global_step: Total training steps completed.
        loss: Current training or validation loss.
        config: Dictionary containing model and training configuration.

    Returns:
        Path to saved checkpoint.
    """
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "global_step": global_step,
        "loss": float(loss),
        "model_state_dict": model.state_dict(),
        "config": config or {},
        "rng_states": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }

    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None and hasattr(scaler, "state_dict"):
        state["scaler_state_dict"] = scaler.state_dict()

    torch.save(state, path)
    return path


def load_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    device: torch.device | str = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load model weights and training state from checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint file.
        model: Target model instance.
        optimizer: Optional optimizer to restore state into.
        scheduler: Optional LR scheduler to restore state into.
        scaler: Optional GradScaler.
        device: Device to map tensors onto.
        strict: Whether model state_dict keys must match strictly.

    Returns:
        Checkpoint dictionary with metadata ('epoch', 'global_step', 'loss', 'config').
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    checkpoint = torch.save if False else torch.load(path, map_location=device)

    # Validate state dict presence
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint at {path} is missing 'model_state_dict'")

    # Load model state
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to load checkpoint into model due to architecture mismatch: {e}"
        ) from e

    # Restore optimizer and scheduler if requested and present
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return checkpoint


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    """Find the latest checkpoint file by modified time in a directory."""
    path = Path(checkpoint_dir)
    if not path.exists():
        return None
    ckpts = list(path.glob("*.pt"))
    if not ckpts:
        return None
    return max(ckpts, key=lambda p: p.stat().st_mtime)


def find_best_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    """Find 'best_model.pt' if it exists, otherwise latest checkpoint."""
    path = Path(checkpoint_dir)
    best = path / "best_model.pt"
    if best.exists():
        return best
    return find_latest_checkpoint(checkpoint_dir)
