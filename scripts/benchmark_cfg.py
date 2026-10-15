"""Generative Quality Benchmark & Classifier-Free Guidance Evaluation Suite.

Evaluates:
1. Independent Classifier Accuracy: Top-1 class accuracy on generated samples across guidance scales.
2. Descriptive Distributional Metrics: Intra-class diversity, spatial contrast, and saturation.
3. Latency & Variability: Warm-up iterations, repeated latency trials, median latency, and variability.
4. Timestep Steering Dynamics: Magnitude of ||eps_cond - eps_uncond|| over diffusion timesteps.
5. All 10 Fashion-MNIST Classes: Full per-class and aggregated summaries.
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
from diffusionlab.evaluation.classifier import load_or_train_classifier
from diffusionlab.evaluation.metrics import evaluate_guidance_vector_dynamics
from diffusionlab.inference.loader import load_model_and_schedule_from_checkpoint


def plot_tradeoff_curves(results: list[dict], output_path: Path):
    """Plots classifier accuracy, descriptive diversity, and saturation across guidance scales."""
    scales = [r["guidance_scale"] for r in results]
    accuracy = [r["aggregated"].get("mean_class_accuracy", 0.0) * 100 for r in results]
    diversity = [r["aggregated"]["mean_diversity"] for r in results]
    saturation = [r["aggregated"]["mean_saturation_rate"] * 100 for r in results]
    contrast = [r["aggregated"]["mean_contrast"] for r in results]

    _fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Independent Classifier Accuracy Plot
    ax1.plot(scales, accuracy, marker="o", color="#2563eb", linewidth=2.5, markersize=7)
    ax1.set_xlabel("Guidance Scale $s$", fontsize=12)
    ax1.set_ylabel("Generated Sample Accuracy (%)", fontsize=12)
    ax1.set_title("Semantic Class Consistency\n(Independent Classifier)", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # 2. Descriptive Intra-Class Diversity Plot
    ax2.plot(scales, diversity, marker="s", color="#7c3aed", linewidth=2.5, markersize=7)
    ax2.set_xlabel("Guidance Scale $s$", fontsize=12)
    ax2.set_ylabel("Intra-Class Diversity (normalized L2)", fontsize=12)
    ax2.set_title("Descriptive Pixel Diversity vs. Scale\n(Higher = greater sample variety)", fontsize=13, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)

    # 3. Descriptive Saturation & Contrast Plot
    color_sat = "#dc2626"
    color_con = "#16a34a"

    ax3.plot(scales, saturation, marker="^", color=color_sat, linewidth=2, label="Saturation Rate (%)")
    ax3.set_xlabel("Guidance Scale $s$", fontsize=12)
    ax3.set_ylabel("Pixel Saturation (|x| > 0.95) %", color=color_sat, fontsize=12)
    ax3.tick_params(axis="y", labelcolor=color_sat)

    ax3_twin = ax3.twinx()
    ax3_twin.plot(scales, contrast, marker="D", color=color_con, linewidth=2, linestyle=":", label="Spatial Contrast")
    ax3_twin.set_ylabel("Spatial Contrast (std)", color=color_con, fontsize=12)
    ax3_twin.tick_params(axis="y", labelcolor=color_con)

    ax3.set_title("Descriptive Contrast & Saturation\n(Monitors activation boundary saturation)", fontsize=13, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Trade-off curve plot saved to: {output_path.resolve()}")


def plot_guidance_dynamics(dynamics: dict[int, float], output_path: Path):
    """Plots the empirical magnitude of ||eps_cond - eps_uncond|| over diffusion timesteps."""
    timesteps = sorted(dynamics.keys())
    norms = [dynamics[t] for t in timesteps]

    plt.figure(figsize=(8, 5))
    plt.plot(timesteps, norms, marker="D", color="#9333ea", linewidth=2.5, markersize=7)
    plt.xlabel("Diffusion Timestep $t$", fontsize=12)
    plt.ylabel(r"Steering Difference Norm $||\epsilon_{cond} - \epsilon_{uncond}||_2 / \sqrt{D}$", fontsize=12)
    plt.title("Measured Steering Difference Across Timesteps", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Steering dynamics plot saved to: {output_path.resolve()}")


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
    parser.add_argument("--num-samples-per-class", type=int, default=16, help="Samples generated per class per scale")
    parser.add_argument(
        "--scales",
        nargs="+",
        type=float,
        default=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        help="Guidance scales to evaluate",
    )
    parser.add_argument("--warmup", type=int, default=1, help="Warm-up trials before timing")
    parser.add_argument("--trials", type=int, default=3, help="Repeated latency trials")
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

    # 1. Independent Classifier Setup & Real Data Validation
    print("\n=== Phase 1: Independent Evaluator Classifier Setup & Real Data Validation ===")
    classifier, provenance = load_or_train_classifier(
        data_dir=data_cfg.get("data_dir", "data"),
        ckpt_path="runs/classifier/fashion_mnist_classifier.pt",
        device=device,
    )
    with open(output_dir / "real_classifier_validation.json", "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
    print(
        f"Verified Independent Classifier Accuracy:\n"
        f"  - Official 10k Test Set (train=False): {provenance['official_test_accuracy']*100:.2f}%\n"
        f"  - 6k Train Validation Split: {provenance['train_val_split_accuracy']*100:.2f}%\n"
    )

    # 2. Guidance Scale Sweep Benchmark Across All 10 Classes
    print(f"=== Phase 2: Guidance Scale Sweep Across All 10 Classes (Steps: {args.steps}) ===")
    benchmark = CFGQualityBenchmark(model, schedule, classifier=classifier, device=device)
    sweep_results = benchmark.run_sweep(
        scales=args.scales,
        classes=tuple(range(10)),
        num_samples_per_class=args.num_samples_per_class,
        steps=args.steps,
        output_dir=output_dir,
        num_warmup=args.warmup,
        num_trials=args.trials,
    )

    # Save detailed JSON containing per-class breakdown
    json_path = output_dir / "cfg_benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=2)

    # Save per-class metrics CSV
    per_class_rows = []
    for scale_res in sweep_results:
        s = scale_res["guidance_scale"]
        for c, m in scale_res["per_class"].items():
            row = {"guidance_scale": s, "class_label": c}
            row.update(m)
            per_class_rows.append(row)

    if per_class_rows:
        per_class_csv = output_dir / "cfg_benchmark_per_class.csv"
        with open(per_class_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=per_class_rows[0].keys())
            writer.writeheader()
            writer.writerows(per_class_rows)

    # 3. Guidance Trade-off Curves Plot
    print("\n=== Phase 3: Generating Trade-Off Visualizations ===")
    plot_tradeoff_curves(sweep_results, output_dir / "cfg_tradeoff_curves.png")

    # 4. Steering Dynamics Across Timesteps
    print("\n=== Phase 4: Evaluating Steering Vector Magnitude Across Timesteps ===")
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

    # 5. Multi-class visual comparison grid with fixed noise
    print("\n=== Phase 5: Fixed-Seed Multi-Class Guidance Progression Grid ===")
    display_classes = [0, 1, 3, 5, 8]  # T-shirt, Trouser, Dress, Sandal, Bag
    comparison_scales = [0.0, 1.0, 2.0, 4.0]
    comparison_samples = []

    for c in display_classes:
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
