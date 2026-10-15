"""Comprehensive evaluation and sample generation script for Classifier-Free Guidance (CFG).

Generates:
1. Individual 16-sample grids for each of the 10 Fashion-MNIST classes.
2. A multi-guidance comparison grid (rows = classes, cols = scales s in [0.0, 1.0, 2.0, 4.0]).
3. Quantitative noise prediction evaluation metrics saved to eval_results.json.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffusionlab.data.datasets import get_dataloaders
from diffusionlab.data.transforms import tensor_to_pil_grid
from diffusionlab.diffusion.forward_process import ForwardDiffusion
from diffusionlab.diffusion.samplers import DDIMSampler
from diffusionlab.evaluation.metrics import (
    count_parameters,
    evaluate_noise_prediction,
    get_checkpoint_size,
)
from diffusionlab.inference.loader import load_model_and_schedule_from_checkpoint

FASHION_MNIST_CLASSES = [
    "t_shirt",
    "trouser",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "ankle_boot",
]


def main():
    parser = argparse.ArgumentParser(description="Evaluate CFG conditional model and generate comparison grids.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="runs/fashion_mnist_conditional/checkpoints/best_model.pt",
        help="Path to conditional model checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/fashion_mnist_conditional/samples",
        help="Output directory for generated sample grids",
    )
    parser.add_argument("--steps", type=int, default=25, help="DDIM sampling steps")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu', 'cuda')")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading conditional checkpoint: {args.checkpoint} onto {device}...")
    model, schedule, raw_ckpt = load_model_and_schedule_from_checkpoint(args.checkpoint, device=device)
    config = raw_ckpt.get("config", {})
    data_cfg = config.get("data", {})

    sampler = DDIMSampler(model, schedule)
    img_size = data_cfg.get("img_size", 28)

    print("\n--- 1. Generating Class-Specific Grids (16 samples per class, s=2.0) ---")
    for class_idx, class_name in enumerate(FASHION_MNIST_CLASSES):
        y = torch.full((16,), class_idx, dtype=torch.long, device=device)
        samples, _ = sampler.sample(
            shape=(16, model.in_channels, img_size, img_size),
            num_inference_steps=args.steps,
            eta=0.0,
            y=y,
            guidance_scale=2.0,
            seed=42 + class_idx,
            device=device,
            show_progress=False,
        )
        grid = tensor_to_pil_grid(samples, nrow=4)
        save_path = output_dir / f"class_{class_idx:02d}_{class_name}.png"
        grid.save(save_path)
        print(f"  Class {class_idx} ({class_name}): saved to {save_path.name}")

    print("\n--- 2. Generating Guidance Scale Comparison (s = 0.0, 1.0, 2.0, 4.0) ---")
    guidance_scales = [0.0, 1.0, 2.0, 4.0]
    comparison_samples = []

    # For each class 0..9, generate 1 sample per guidance scale using identical fixed initial noise
    for class_idx in range(10):
        y = torch.full((1,), class_idx, dtype=torch.long, device=device)
        fixed_noise_seed = 1000 + class_idx
        for s in guidance_scales:
            sample, _ = sampler.sample(
                shape=(1, model.in_channels, img_size, img_size),
                num_inference_steps=args.steps,
                eta=0.0,
                y=y,
                guidance_scale=s,
                seed=fixed_noise_seed,
                device=device,
                show_progress=False,
            )
            comparison_samples.append(sample)

    comparison_tensor = torch.cat(comparison_samples, dim=0)  # Shape: (40, C, H, W)
    comp_grid = tensor_to_pil_grid(comparison_tensor, nrow=len(guidance_scales))
    comp_path = output_dir / "cfg_scale_comparison.png"
    comp_grid.save(comp_path)
    print(f"Guidance scale comparison grid saved to: {comp_path.resolve()}")

    print("\n--- 3. Quantitative Validation Error ---")
    forward_diff = ForwardDiffusion(schedule)
    _, val_loader = get_dataloaders(
        dataset_name=data_cfg.get("dataset_name", "fashion_mnist"),
        data_dir=data_cfg.get("data_dir", "data"),
        img_size=img_size,
        batch_size=64,
        val_split=0.1,
        num_workers=0,
        subset_size=data_cfg.get("subset_size", None),
        use_synthetic_fallback=True,
    )

    metrics = evaluate_noise_prediction(
        model=model,
        forward_diff=forward_diff,
        data_loader=val_loader,
        device=device,
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
        "guidance_scales_evaluated": guidance_scales,
        "classes_evaluated": FASHION_MNIST_CLASSES,
        "device": str(device),
    }

    print("\n--- Quantitative Evaluation Summary ---")
    for k, v in results.items():
        print(f"  {k}: {v}")

    eval_file = Path(args.checkpoint).parent / "eval_results.json"
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation metrics to: {eval_file}")


if __name__ == "__main__":
    main()
