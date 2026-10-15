"""Orchestration script for controlled scientific experiments (Phase 7)."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffusionlab.data.datasets import get_dataloaders
from diffusionlab.data.transforms import tensor_to_pil_grid
from diffusionlab.diffusion.forward_process import ForwardDiffusion
from diffusionlab.diffusion.samplers import DDIMSampler
from diffusionlab.diffusion.schedules import get_beta_schedule
from diffusionlab.models.unet import UNet
from diffusionlab.training.trainer import DiffusionTrainer


def run_experiment_a_noise_schedules(device, output_dir: Path, steps_per_run: int = 50):
    """Experiment A: Compare Linear vs Cosine Beta Schedules."""
    print("\n=== Running Experiment A: Linear vs Cosine Noise Schedules ===")
    exp_dir = output_dir / "exp_a_schedules"
    exp_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    train_loader, val_loader = get_dataloaders(
        dataset_name="fashion_mnist",
        batch_size=32,
        subset_size=256,
        use_synthetic_fallback=True,
    )

    for sched_name in ["linear", "cosine"]:
        print(f"Testing schedule: {sched_name.upper()}...")
        schedule = get_beta_schedule(sched_name, timesteps=200, device=device)
        forward_diff = ForwardDiffusion(schedule)
        model = UNet(in_channels=1, base_channels=16, channel_multipliers=(1, 2), num_res_blocks=1).to(device)

        config = {
            "epochs": 1,
            "lr": 5e-4,
            "run_dir": str(exp_dir / sched_name),
            "log_interval": 10,
        }
        trainer = DiffusionTrainer(
            model=model,
            forward_diff=forward_diff,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
        )

        history = trainer.fit(smoke_test=False)

        # Generate sample grid
        sampler = DDIMSampler(model, schedule)
        samples, _ = sampler.sample(shape=(4, 1, 28, 28), num_inference_steps=20, device=device, seed=42)
        grid = tensor_to_pil_grid(samples)
        grid.save(exp_dir / f"samples_{sched_name}.png")

        results[sched_name] = {
            "train_loss": history["train_loss"][-1],
            "val_loss": history["val_loss"][-1],
            "duration_s": history["total_duration"],
        }

    # Plot comparison
    fig, ax = plt.subplots(figsize=(6, 4))
    scheds = list(results.keys())
    val_losses = [results[s]["val_loss"] for s in scheds]
    ax.bar(scheds, val_losses, color=["#4285F4", "#34A853"])
    ax.set_ylabel("Validation Noise-MSE Loss")
    ax.set_title("Experiment A: Linear vs Cosine Schedule Loss")
    plt.tight_layout()
    fig.savefig(exp_dir / "schedule_loss_comparison.png")
    plt.close(fig)

    with open(exp_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Experiment A completed. Results saved to: {exp_dir}")


def run_experiment_b_unet_capacity(device, output_dir: Path):
    """Experiment B: Model Capacity (Compact vs Medium)."""
    print("\n=== Running Experiment B: U-Net Network Capacity ===")
    exp_dir = output_dir / "exp_b_capacity"
    exp_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    train_loader, val_loader = get_dataloaders(
        dataset_name="fashion_mnist",
        batch_size=32,
        subset_size=256,
        use_synthetic_fallback=True,
    )
    schedule = get_beta_schedule("cosine", timesteps=100, device=device)
    forward_diff = ForwardDiffusion(schedule)

    configs = {
        "compact": {"base_channels": 16, "channel_multipliers": (1, 2), "num_res_blocks": 1},
        "medium": {"base_channels": 32, "channel_multipliers": (1, 2), "num_res_blocks": 1},
    }

    for cap_name, model_args in configs.items():
        print(f"Testing model capacity: {cap_name.upper()}...")
        model = UNet(in_channels=1, **model_args).to(device)
        num_params = model.num_parameters()

        config = {
            "epochs": 1,
            "lr": 5e-4,
            "run_dir": str(exp_dir / cap_name),
            "log_interval": 10,
        }
        trainer = DiffusionTrainer(
            model=model,
            forward_diff=forward_diff,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
        )

        history = trainer.fit(smoke_test=False)

        results[cap_name] = {
            "parameters": num_params,
            "train_loss": history["train_loss"][-1],
            "val_loss": history["val_loss"][-1],
            "duration_s": history["total_duration"],
        }

    # Plot comparison
    fig, ax1 = plt.subplots(figsize=(6, 4))
    caps = list(results.keys())
    params = [results[c]["parameters"] for c in caps]
    val_losses = [results[c]["val_loss"] for c in caps]

    ax1.bar(caps, params, alpha=0.6, color="#EA4335", label="Parameters")
    ax1.set_ylabel("Parameters", color="#EA4335")
    ax2 = ax1.twinx()
    ax2.plot(caps, val_losses, color="#185ABC", marker="o", linewidth=2, label="Val Loss")
    ax2.set_ylabel("Validation Loss", color="#185ABC")
    ax1.set_title("Experiment B: U-Net Capacity vs Loss")
    plt.tight_layout()
    fig.savefig(exp_dir / "capacity_comparison.png")
    plt.close(fig)

    with open(exp_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Experiment B completed. Results saved to: {exp_dir}")


def run_experiment_c_sampling_efficiency(device, output_dir: Path):
    """Experiment C: Sampling Efficiency across inference step budgets."""
    print("\n=== Running Experiment C: DDPM vs DDIM Sampling Efficiency ===")
    exp_dir = output_dir / "exp_c_sampling"
    exp_dir.mkdir(parents=True, exist_ok=True)

    schedule = get_beta_schedule("linear", timesteps=100, device=device)
    model = UNet(in_channels=1, base_channels=16, channel_multipliers=(1, 2), num_res_blocks=1).to(device)

    from diffusionlab.evaluation.benchmarks import SamplingBenchmark

    benchmark = SamplingBenchmark(model, schedule, device=device)
    results = benchmark.run_suite(
        step_counts=[100, 50, 20, 10, 5],
        batch_size=4,
        output_dir=exp_dir,
    )

    # Plot Latency vs Steps
    ddim_results = [r for r in results if r["method"] == "DDIM"]
    steps = [r["steps"] for r in ddim_results]
    latencies = [r["avg_batch_latency_s"] for r in ddim_results]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, latencies, marker="s", color="#FBBC04", linewidth=2, label="DDIM (eta=0)")
    # Add DDPM baseline point
    ddpm_r = next(r for r in results if r["method"] == "DDPM")
    ax.axhline(y=ddpm_r["avg_batch_latency_s"], color="red", linestyle="--", label="DDPM (100 steps)")

    ax.set_xlabel("Inference Steps")
    ax.set_ylabel("Batch Latency (seconds)")
    ax.set_title("Experiment C: Sampling Latency vs Steps")
    ax.legend()
    plt.tight_layout()
    fig.savefig(exp_dir / "sampling_latency_vs_steps.png")
    plt.close(fig)

    print(f"Experiment C completed. Results saved to: {exp_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run controlled diffusion experiments.")
    parser.add_argument("--experiments", nargs="+", default=["A", "B", "C"], help="Experiments to run: A, B, C")
    parser.add_argument("--output-dir", type=str, default="reports/experiments", help="Output directory")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu', 'cuda')")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running Controlled Experiments Suite on {device}...")

    if "A" in args.experiments:
        run_experiment_a_noise_schedules(device, out_dir)
    if "B" in args.experiments:
        run_experiment_b_unet_capacity(device, out_dir)
    if "C" in args.experiments:
        run_experiment_c_sampling_efficiency(device, out_dir)

    print(f"\nAll requested experiments finished! Check reports under: {out_dir}")


if __name__ == "__main__":
    main()
