"""Generative Quality Benchmark & Classifier-Free Guidance Evaluation Suite.

Evaluates:
1. Guidance Scale Sweep: Intra-class diversity, contrast, saturation, and latency across scales.
2. Steering Vector Dynamics: Magnitude of ||eps_cond - eps_uncond|| over diffusion timesteps.
3. Visualization: Trade-off curves and steering dynamics plots.
4. Comprehensive machine-readable metrics (JSON and CSV).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffusionlab.data.datasets import get_dataloaders
from diffusionlab.data.transforms import tensor_to_pil_grid
from diffusionlab.diffusion.forward_process import ForwardDiffusion
from diffusionlab.evaluation.benchmarks import CFGQualityBenchmark
from diffusionlab.evaluation.metrics import evaluate_guidance_vector_dynamics
from diffusionlab.inference.loader import load_model_and_schedule_from_checkpoint


def plot_tradeoff_curves(results: list[dict], output_path: Path):
    """Plot intra-class diversity and pixel saturation rate as a function of guidance scale."""
    scales = [r["guidance_scale"] for r in results]
    diversity = [r["diversity"] for r in results]
    saturation = [r["saturation_rate"] * 100 for r in results]
    contrast = [r["mean_contrast"] for r in results]

    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Diversity plot
    ax1.plot(scales, diversity, marker="o", color="#2563eb", linewidth=2, markersize=6)
    ax1.set_xlabel("Guidance Scale $s$", fontsize=12)
    ax1.set_ylabel("Intra-Class Diversity (normalized L2)", fontsize=12)
    ax1.set_title("Sample Diversity vs. Guidance Scale", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Saturation & Contrast plot
    color_sat = "#dc2626"
    color_con = "#16a34a"

    ax2.plot(scales, saturation, marker="s", color=color_sat, linewidth=2, label="Saturation Rate (%)")
    ax2.set_xlabel("Guidance Scale $s$", fontsize=12)
    ax2.set_ylabel("Pixel Saturation (|x| > 0.95) %", color=color_sat, fontsize=12)
    ax2.tick_params(axis="y", labelcolor=color_sat)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(scales, contrast, marker="^", color=color_con, linewidth=2, linestyle=":", label="Spatial Contrast")
    ax2_twin.set_ylabel("Spatial Contrast (pixel std)", color=color_con, fontsize=12)
    ax2_twin.tick_params(axis="y", labelcolor=color_con)

    ax2.set_title("Contrast & Saturation Trade-Off", fontsize=13, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Trade-off curve plot saved to: {output_path.resolve()}")


def plot_guidance_dynamics(dynamics: dict[int, float], output_path: Path):
    """Plot guidance steering vector norm ||eps_cond - eps_uncond|| over diffusion timesteps."""
    timesteps = sorted(dynamics.keys())
    norms = [dynamics[t] for t in timesteps]

    plt.figure(figsize=(8, 5))
    plt.plot(timesteps, norms, marker="D", color="#9333ea", linewidth=2.5, markersize=7)
    plt.xlabel("Diffusion Timestep $t$", fontsize=12)
    plt.ylabel(r"Guidance Vector Magnitude $||\epsilon_{cond} - \epsilon_{uncond}||_2$", fontsize=12)
    plt.title("CFG Steering Vector Dynamics Across Timesteps", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Guidance dynamics plot saved to: {output_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive CFG Evaluation and Generative Benchmark.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="runs/fashion_mnist_conditional/checkpoints/best_model.pt",
        help="Path to trained conditional checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/experiments/exp_e_cfg_evaluation",
        help="Destination directory for benchmark reports and figures",
    )
    parser.add_argument("--steps", type=int, default=25, help="DDIM sampling steps")
    parser.add_argument("--num-samples", type=int, default=16, help="Samples per scale evaluation")
    parser.add_argument(
        "--scales",
        nargs="+",
        type=float,
        default=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        help="Guidance scales to evaluate",
    )
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu', 'cuda')")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint} onto {device}...")
    model, schedule, raw_ckpt = load_model_and_schedule_from_checkpoint(args.checkpoint, device=device)
    config = raw_ckpt.get("config", {})
    data_cfg = config.get("data", {})
    forward_diff = ForwardDiffusion(schedule)

    # 1. Guidance Scale Sweep Benchmark
    print("\n=== Phase 1: Guidance Scale Generative Quality Sweep ===")
    benchmark = CFGQualityBenchmark(model, schedule, device=device)
    sweep_results = benchmark.run_sweep(
        scales=args.scales,
        class_label=0,  # T-shirt
        num_samples=args.num_samples,
        steps=args.steps,
        output_dir=output_dir,
    )

    # Copy sweep results to output_dir
    json_path = output_dir / "cfg_benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=2)

    csv_path = output_dir / "cfg_benchmark_results.csv"
    if sweep_results:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sweep_results[0].keys())
            writer.writeheader()
            writer.writerows(sweep_results)

    # 2. Guidance Trade-off Curves Plot
    print("\n=== Phase 2: Generating Trade-Off Visualizations ===")
    plot_tradeoff_curves(sweep_results, output_dir / "cfg_tradeoff_curves.png")

    # 3. Steering Dynamics Across Timesteps
    print("\n=== Phase 3: Evaluating Steering Vector Dynamics Across Timesteps ===")
    _, val_loader = get_dataloaders(
        dataset_name=data_cfg.get("dataset_name", "fashion_mnist"),
        data_dir=data_cfg.get("data_dir", "data"),
        img_size=data_cfg.get("img_size", 28),
        batch_size=64,
        val_split=0.1,
        num_workers=0,
        subset_size=data_cfg.get("subset_size", None),
        use_synthetic_fallback=True,
    )

    sample_timesteps = [10, 30, 50, 75, 100, 130, 160, 185]
    valid_timesteps = [t for t in sample_timesteps if t < schedule.timesteps]

    dynamics = evaluate_guidance_vector_dynamics(
        model=model,
        forward_diff=forward_diff,
        data_loader=val_loader,
        device=device,
        timesteps=valid_timesteps,
        max_batches=5,
    )

    dynamics_path = output_dir / "guidance_dynamics.json"
    with open(dynamics_path, "w", encoding="utf-8") as f:
        json.dump(dynamics, f, indent=2)

    plot_guidance_dynamics(dynamics, output_dir / "guidance_dynamics_over_timesteps.png")

    # 4. Multi-class visual comparison grid with fixed noise
    print("\n=== Phase 4: Fixed-Seed Multi-Class Guidance Progression Grid ===")
    classes = [0, 1, 3, 5, 8]  # T-shirt, Trouser, Dress, Sandal, Bag
    comparison_scales = [0.0, 1.0, 2.0, 4.0]
    comparison_samples = []

    for c in classes:
        y = torch.full((1,), c, dtype=torch.long, device=device)
        for s in comparison_scales:
            sample, _ = benchmark.sampler.sample(
                shape=(1, model.in_channels, 28, 28),
                num_inference_steps=args.steps,
                eta=0.0,
                y=y,
                guidance_scale=s,
                seed=5000 + c,
                device=device,
                show_progress=False,
            )
            comparison_samples.append(sample)

    progression_grid = tensor_to_pil_grid(torch.cat(comparison_samples, dim=0), nrow=len(comparison_scales))
    grid_path = output_dir / "guidance_progression_grid.png"
    progression_grid.save(grid_path)
    print(f"Guidance progression grid saved to: {grid_path.resolve()}")

    print(f"\nAll benchmark artifacts successfully written to: {output_dir.resolve()}\n")


if __name__ == "__main__":
    main()
