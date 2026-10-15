"""Evaluation script for quantitative analysis of trained checkpoints."""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffusionlab.data.datasets import get_dataloaders
from diffusionlab.diffusion.forward_process import ForwardDiffusion
from diffusionlab.evaluation.metrics import (
    count_parameters,
    evaluate_noise_prediction,
    get_checkpoint_size,
)
from diffusionlab.inference.loader import load_model_and_schedule_from_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained DiffusionLab model checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing dataset")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--max-batches", type=int, default=None, help="Max batches to evaluate")
    parser.add_argument("--output", type=str, default=None, help="Path to save evaluation metrics JSON")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu', 'cuda')")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Loading checkpoint: {args.checkpoint} on {device}...")

    model, schedule, raw_ckpt = load_model_and_schedule_from_checkpoint(args.checkpoint, device=device)
    forward_diff = ForwardDiffusion(schedule)
    data_cfg = raw_ckpt.get("config", {}).get("data", {})

    # Load validation data
    _, val_loader = get_dataloaders(
        dataset_name=data_cfg.get("dataset_name", "fashion_mnist"),
        data_dir=args.data_dir,
        img_size=data_cfg.get("img_size", 28),
        batch_size=args.batch_size,
        val_split=0.1,
        num_workers=0,
        subset_size=data_cfg.get("subset_size", None),
        use_synthetic_fallback=True,
    )

    print("Evaluating noise prediction error on validation dataset...")
    metrics = evaluate_noise_prediction(
        model=model,
        forward_diff=forward_diff,
        data_loader=val_loader,
        device=device,
        max_batches=args.max_batches,
    )

    param_stats = count_parameters(model)
    ckpt_stats = get_checkpoint_size(args.checkpoint)

    results = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "epoch": raw_ckpt.get("epoch", 0),
        "global_step": raw_ckpt.get("global_step", 0),
        "val_mse_loss": round(metrics["mse_loss"], 6),
        "val_l1_loss": round(metrics["l1_loss"], 6),
        "trainable_parameters": param_stats["trainable_parameters"],
        "checkpoint_size_mb": round(ckpt_stats["size_mb"], 2),
        "device": str(device),
    }

    print("\n--- Quantitative Evaluation Results ---")
    for k, v in results.items():
        print(f"  {k}: {v}")

    out_file = args.output or str(Path(args.checkpoint).parent / "eval_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation results to: {out_file}\n")


if __name__ == "__main__":
    main()
