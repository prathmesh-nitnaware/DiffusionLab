"""Benchmarking script for comparing DDPM and DDIM inference performance."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffusionlab.diffusion.schedules import get_beta_schedule
from diffusionlab.evaluation.benchmarks import SamplingBenchmark
from diffusionlab.inference.loader import load_model_and_schedule_from_checkpoint
from diffusionlab.models.unet import UNet


def main():
    parser = argparse.ArgumentParser(description="Benchmark sampling latency and throughput for DDPM and DDIM.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint to load model weights from")
    parser.add_argument("--steps", nargs="+", type=int, default=[1000, 250, 100, 50, 20], help="DDIM step counts")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for benchmarking")
    parser.add_argument("--timesteps", type=int, default=1000, help="Total diffusion steps in schedule")
    parser.add_argument("--base-channels", type=int, default=32, help="U-Net base channels")
    parser.add_argument("--output-dir", type=str, default="runs/benchmarks", help="Output directory for reports")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu', 'cuda')")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    if args.checkpoint:
        print(f"Loading model from checkpoint: {args.checkpoint}...")
        model, schedule, _ = load_model_and_schedule_from_checkpoint(args.checkpoint, device=device)
    else:
        print("Benchmarking with freshly initialized model architecture...")
        schedule = get_beta_schedule("linear", timesteps=args.timesteps, device=device)
        model = UNet(
            in_channels=1,
            out_channels=1,
            base_channels=args.base_channels,
            channel_multipliers=(1, 2),
            num_res_blocks=1,
            dropout=0.0,
        ).to(device)

    benchmark = SamplingBenchmark(model, schedule, device=device)
    benchmark.run_suite(
        step_counts=args.steps,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
