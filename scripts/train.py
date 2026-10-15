"""Training entrypoint for DiffusionLab."""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch import optim

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffusionlab.data.datasets import get_dataloaders
from diffusionlab.diffusion.forward_process import ForwardDiffusion
from diffusionlab.diffusion.schedules import get_beta_schedule
from diffusionlab.models.unet import UNet
from diffusionlab.training.trainer import DiffusionTrainer


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train a DDPM model on Fashion-MNIST or synthetic data.")
    parser.add_argument("--config", type=str, default="configs/fashion_mnist.yaml", help="Path to YAML config file")
    parser.add_argument("--smoke-test", action="store_true", help="Run in fast smoke-test mode (1 batch)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu', 'cuda', etc.)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    config = load_yaml(args.config)

    # CLI Overrides
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["lr"] = args.lr

    model_cfg = config.get("model", {})
    diff_cfg = config.get("diffusion", {})
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})

    # Seed
    seed = args.seed if args.seed is not None else train_cfg.get("seed", 42)
    torch.manual_seed(seed)

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== DiffusionLab Training ===")
    print(f"Configuration: {args.config}")
    print(f"Device: {device} (CUDA available: {torch.cuda.is_available()})")
    print(f"Reproducibility Seed: {seed}")

    # 1. Datasets and Loaders
    train_loader, val_loader = get_dataloaders(
        dataset_name=data_cfg.get("dataset_name", "fashion_mnist"),
        data_dir=data_cfg.get("data_dir", "data"),
        img_size=data_cfg.get("img_size", 28),
        batch_size=data_cfg.get("batch_size", 64),
        val_split=data_cfg.get("val_split", 0.1),
        num_workers=data_cfg.get("num_workers", 0),
        subset_size=data_cfg.get("subset_size", None),
        use_synthetic_fallback=data_cfg.get("use_synthetic_fallback", False),
    )

    # 2. Diffusion Schedule & Forward Process
    schedule = get_beta_schedule(
        schedule_type=diff_cfg.get("schedule_type", "cosine"),
        timesteps=diff_cfg.get("timesteps", 1000),
        beta_start=diff_cfg.get("beta_start", 1e-4),
        beta_end=diff_cfg.get("beta_end", 0.02),
        cosine_s=diff_cfg.get("cosine_s", 0.008),
        device=device,
    )
    forward_diff = ForwardDiffusion(schedule)

    # 3. Model
    model = UNet(
        in_channels=model_cfg.get("in_channels", 1),
        out_channels=model_cfg.get("out_channels", 1),
        base_channels=model_cfg.get("base_channels", 32),
        channel_multipliers=model_cfg.get("channel_multipliers", (1, 2, 4)),
        num_res_blocks=model_cfg.get("num_res_blocks", 2),
        attention_resolutions=model_cfg.get("attention_resolutions", (2,)),
        dropout=model_cfg.get("dropout", 0.0),
        num_classes=model_cfg.get("num_classes", None),
    ).to(device)

    print(f"Model instantiated with {model.num_parameters():,} trainable parameters.")

    # 4. Optimizer & Scheduler
    lr = train_cfg.get("lr", 2e-4)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=train_cfg.get("weight_decay", 1e-4))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg.get("epochs", 10), eta_min=1e-6)

    full_config = {
        "model": model_cfg,
        "diffusion": diff_cfg,
        "data": data_cfg,
        "training": train_cfg,
    }

    # 5. Trainer
    trainer = DiffusionTrainer(
        model=model,
        forward_diff=forward_diff,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=full_config,
        device=device,
        p_uncond=train_cfg.get("p_uncond", 0.1),
        seed=seed,
    )

    # Resume from checkpoint if requested
    if args.resume:
        print(f"Resuming training from checkpoint: {args.resume}")
        trainer.resume_from_checkpoint(args.resume, restore_rng=True)

    # Execute training or smoke test
    history = trainer.fit(smoke_test=args.smoke_test)
    print("Training run finished successfully.")
    print(f"Checkpoints saved to: {trainer.checkpoint_dir}")
    print(f"Samples saved to: {trainer.sample_dir}")
    return history


if __name__ == "__main__":
    main()
