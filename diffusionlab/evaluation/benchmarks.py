"""Benchmarking suite for diffusion sampling strategies."""

import csv
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from diffusionlab.data.transforms import tensor_to_pil_grid
from diffusionlab.diffusion.samplers import DDIMSampler, DDPMSampler
from diffusionlab.diffusion.schedules import BetaSchedule
from diffusionlab.evaluation.metrics import get_memory_stats


class SamplingBenchmark:
    """Profiles sampling duration, latency per image, throughput, and memory consumption."""

    def __init__(
        self,
        model: nn.Module,
        schedule: BetaSchedule,
        device: torch.device | str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.schedule = schedule
        self.schedule.to(self.device)

        self.ddpm = DDPMSampler(self.model, self.schedule)
        self.ddim = DDIMSampler(self.model, self.schedule)

    @torch.no_grad()
    def benchmark_sampler(
        self,
        method: str = "ddim",
        steps: int = 50,
        batch_size: int = 4,
        img_size: int = 28,
        eta: float = 0.0,
        num_warmup: int = 1,
        num_runs: int = 2,
    ) -> dict[str, Any]:
        """Run benchmark for a specific sampler configuration."""
        shape = (batch_size, self.model.in_channels, img_size, img_size)

        # Warmup
        for _ in range(num_warmup):
            if method.lower() == "ddpm":
                self.ddpm.sample(shape=shape, device=self.device, show_progress=False)
            else:
                self.ddim.sample(
                    shape=shape,
                    num_inference_steps=min(steps, self.schedule.timesteps),
                    eta=eta,
                    device=self.device,
                    show_progress=False,
                )

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)

        run_times = []
        samples = None

        for _ in range(num_runs):
            start = time.perf_counter()

            if method.lower() == "ddpm":
                samples, _ = self.ddpm.sample(shape=shape, device=self.device, show_progress=False)
            else:
                samples, _ = self.ddim.sample(
                    shape=shape,
                    num_inference_steps=min(steps, self.schedule.timesteps),
                    eta=eta,
                    device=self.device,
                    show_progress=False,
                )

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

            run_times.append(time.perf_counter() - start)

        avg_latency_batch_sec = sum(run_times) / len(run_times)
        latency_per_image_ms = (avg_latency_batch_sec / batch_size) * 1000.0
        throughput_img_per_sec = batch_size / avg_latency_batch_sec

        memory_stats = get_memory_stats(self.device)

        return {
            "method": method.upper(),
            "steps": steps,
            "batch_size": batch_size,
            "eta": eta if method.lower() == "ddim" else 1.0,
            "avg_batch_latency_s": round(avg_latency_batch_sec, 4),
            "latency_per_image_ms": round(latency_per_image_ms, 2),
            "throughput_images_per_s": round(throughput_img_per_sec, 3),
            "max_memory_mb": memory_stats["max_allocated_mb"],
            "device": str(self.device),
            "last_samples": samples,
        }

    def run_suite(
        self,
        step_counts: Sequence[int] = (1000, 250, 100, 50, 20),
        batch_size: int = 4,
        img_size: int = 28,
        output_dir: str | Path = "runs/benchmarks",
        save_grids: bool = True,
    ) -> list[dict[str, Any]]:
        """Run full comparison suite between DDPM and DDIM."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        results = []

        print(f"\n--- Running Sampling Benchmark (Device: {self.device}) ---")

        # 1. DDPM Benchmark
        print(f"Benchmarking DDPM (full {self.schedule.timesteps} steps)...")
        ddpm_res = self.benchmark_sampler(
            method="ddpm",
            steps=self.schedule.timesteps,
            batch_size=batch_size,
            img_size=img_size,
        )
        if save_grids and ddpm_res["last_samples"] is not None:
            grid = tensor_to_pil_grid(ddpm_res["last_samples"])
            grid.save(out_path / "sample_grid_ddpm.png")
        ddpm_res_clean = {k: v for k, v in ddpm_res.items() if k != "last_samples"}
        results.append(ddpm_res_clean)
        print(
            f"  DDPM: {ddpm_res['avg_batch_latency_s']:.2f}s/batch | "
            f"{ddpm_res['latency_per_image_ms']:.1f}ms/img | "
            f"{ddpm_res['throughput_images_per_s']:.2f} img/s"
        )

        # 2. DDIM across step counts
        for steps in step_counts:
            if steps > self.schedule.timesteps:
                continue
            print(f"Benchmarking DDIM ({steps} steps, eta=0.0)...")
            ddim_res = self.benchmark_sampler(
                method="ddim",
                steps=steps,
                batch_size=batch_size,
                img_size=img_size,
                eta=0.0,
            )
            if save_grids and ddim_res["last_samples"] is not None:
                grid = tensor_to_pil_grid(ddim_res["last_samples"])
                grid.save(out_path / f"sample_grid_ddim_{steps}steps.png")
            ddim_res_clean = {k: v for k, v in ddim_res.items() if k != "last_samples"}
            results.append(ddim_res_clean)
            print(
                f"  DDIM-{steps}: {ddim_res['avg_batch_latency_s']:.2f}s/batch | "
                f"{ddim_res['latency_per_image_ms']:.1f}ms/img | "
                f"{ddim_res['throughput_images_per_s']:.2f} img/s"
            )

        # Save JSON
        json_path = out_path / "benchmark_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        # Save CSV
        csv_path = out_path / "benchmark_results.csv"
        if results:
            keys = results[0].keys()
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(results)

        print(f"Benchmark results saved to: {json_path} and {csv_path}\n")
        return results


class CFGQualityBenchmark:
    """Benchmark suite for Classifier-Free Guidance (CFG).

    Profiles generative quality metrics (intra-class diversity, spatial contrast,
    saturation rate) and computational efficiency (throughput, batch latency, memory)
    across a configurable spectrum of guidance scales and sampling step budgets.
    """

    def __init__(
        self,
        model: nn.Module,
        schedule: BetaSchedule,
        device: torch.device | str = "cpu",
    ):
        from diffusionlab.evaluation.metrics import (
            compute_contrast_and_saturation,
            compute_intra_class_diversity,
        )

        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.schedule = schedule
        self.schedule.to(self.device)
        self.sampler = DDIMSampler(self.model, self.schedule)

        self._compute_diversity = compute_intra_class_diversity
        self._compute_contrast = compute_contrast_and_saturation

    @torch.no_grad()
    def evaluate_scale(
        self,
        guidance_scale: float,
        class_label: int = 0,
        num_samples: int = 16,
        steps: int = 25,
        img_size: int = 28,
        eta: float = 0.0,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Evaluates generative quality metrics and latency for a single guidance scale."""
        shape = (num_samples, self.model.in_channels, img_size, img_size)
        y = torch.full((num_samples,), class_label, dtype=torch.long, device=self.device)

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)

        start_time = time.perf_counter()
        samples, _ = self.sampler.sample(
            shape=shape,
            num_inference_steps=min(steps, self.schedule.timesteps),
            eta=eta,
            y=y,
            guidance_scale=guidance_scale,
            seed=seed,
            device=self.device,
            show_progress=False,
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        total_time_s = time.perf_counter() - start_time
        latency_per_img_ms = (total_time_s / num_samples) * 1000.0
        throughput_img_s = num_samples / max(1e-6, total_time_s)

        diversity = self._compute_diversity(samples)
        contrast_stats = self._compute_contrast(samples)
        memory_stats = get_memory_stats(self.device)

        return {
            "guidance_scale": float(guidance_scale),
            "class_label": int(class_label),
            "steps": int(steps),
            "num_samples": int(num_samples),
            "diversity": diversity,
            "mean_contrast": contrast_stats["mean_contrast"],
            "saturation_rate": contrast_stats["saturation_rate"],
            "dynamic_range": contrast_stats["dynamic_range"],
            "batch_latency_s": round(total_time_s, 4),
            "latency_per_image_ms": round(latency_per_img_ms, 2),
            "throughput_images_per_s": round(throughput_img_s, 3),
            "max_memory_mb": memory_stats["max_allocated_mb"],
            "samples": samples,
        }

    def run_sweep(
        self,
        scales: Sequence[float] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
        class_label: int = 0,
        num_samples: int = 16,
        steps: int = 25,
        img_size: int = 28,
        output_dir: str | Path = "runs/benchmarks/cfg",
        seed: int = 42,
        save_grids: bool = True,
    ) -> list[dict[str, Any]]:
        """Run full evaluation sweep across guidance scales."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        results = []

        print(f"\n--- Running CFG Guidance Scale Sweep (Class {class_label}, Steps: {steps}, Seed: {seed}) ---")

        for s in scales:
            res = self.evaluate_scale(
                guidance_scale=s,
                class_label=class_label,
                num_samples=num_samples,
                steps=steps,
                img_size=img_size,
                seed=seed,
            )

            if save_grids and res["samples"] is not None:
                grid = tensor_to_pil_grid(res["samples"], nrow=int(num_samples**0.5))
                grid.save(out_path / f"cfg_scale_{s:.1f}.png")

            clean_res = {k: v for k, v in res.items() if k != "samples"}
            results.append(clean_res)
            print(
                f"  Scale {s:4.1f}: Diversity={res['diversity']:.4f} | "
                f"Contrast={res['mean_contrast']:.4f} | "
                f"Saturation={res['saturation_rate']*100:.1f}% | "
                f"Throughput={res['throughput_images_per_s']:.2f} img/s"
            )

        # Save JSON
        json_path = out_path / "cfg_benchmark_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        # Save CSV
        csv_path = out_path / "cfg_benchmark_results.csv"
        if results:
            keys = results[0].keys()
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(results)

        print(f"CFG Benchmark results saved to: {json_path} and {csv_path}\n")
        return results

