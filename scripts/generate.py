"""Inference and image generation script for DiffusionLab."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffusionlab.data.transforms import tensor_to_pil_grid
from diffusionlab.diffusion.samplers import DDIMSampler, DDPMSampler
from diffusionlab.inference.loader import load_model_and_schedule_from_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Generate images from a trained DiffusionLab checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--sampler", type=str, choices=["ddpm", "ddim"], default="ddim", help="Sampling method")
    parser.add_argument("--steps", type=int, default=50, help="Inference steps for DDIM")
    parser.add_argument("--eta", type=float, default=0.0, help="Stochasticity eta for DDIM (0.0 = deterministic ODE)")
    parser.add_argument("--num-samples", type=int, default=16, help="Number of images to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")
    parser.add_argument("--output", type=str, default="generated_samples.png", help="Path to save output grid")
    parser.add_argument("--save-trajectory", action="store_true", help="Save intermediate denoising trajectory")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu', 'cuda')")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Loading checkpoint: {args.checkpoint} onto {device}...")

    model, schedule, raw_ckpt = load_model_and_schedule_from_checkpoint(args.checkpoint, device=device)
    config = raw_ckpt.get("config", {})

    img_size = config.get("data", {}).get("img_size", 28)
    shape = (args.num_samples, model.in_channels, img_size, img_size)

    print(
        f"Generating {args.num_samples} images using {args.sampler.upper()} "
        f"(Steps: {args.steps if args.sampler == 'ddim' else schedule.timesteps}, Seed: {args.seed})..."
    )

    if args.sampler == "ddpm":
        sampler = DDPMSampler(model, schedule)
        samples, intermediates = sampler.sample(
            shape=shape,
            device=device,
            seed=args.seed,
            save_intermediate_steps=schedule.timesteps // 8 if args.save_trajectory else None,
            show_progress=True,
        )
    else:
        sampler = DDIMSampler(model, schedule)
        samples, intermediates = sampler.sample(
            shape=shape,
            num_inference_steps=args.steps,
            eta=args.eta,
            device=device,
            seed=args.seed,
            save_intermediate=args.save_trajectory,
            show_progress=True,
        )

    # Save final grid
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil_grid = tensor_to_pil_grid(samples, nrow=int(args.num_samples**0.5))
    pil_grid.save(out_path)
    print(f"Generated sample grid saved to: {out_path.resolve()}")

    # Save trajectory if requested
    if args.save_trajectory and len(intermediates) > 1:
        traj_tensor = torch.cat(intermediates, dim=0)
        traj_grid = tensor_to_pil_grid(traj_tensor, nrow=args.num_samples)
        traj_path = out_path.parent / f"trajectory_{out_path.name}"
        traj_grid.save(traj_path)
        print(f"Denoising trajectory saved to: {traj_path.resolve()}")


if __name__ == "__main__":
    main()
