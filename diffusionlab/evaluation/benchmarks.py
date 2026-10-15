"""Benchmarking suite for diffusion sampling strategies."""

import csv
import json
import statistics
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

    Profiles:
    1. Independent classifier accuracy on generated samples across guidance scales.
    2. Descriptive pixel-space distributions (intra-class diversity, contrast, saturation).
    3. Computational efficiency (warm-up iterations, repeated trials, median latency, variability).
    4. Per-class metrics and aggregated macro summaries across all 10 Fashion-MNIST classes.
    """

    def __init__(
        self,
        model: nn.Module,
        schedule: BetaSchedule,
        classifier: nn.Module | None = None,
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
        self.classifier = classifier.to(self.device).eval() if classifier is not None else None

        self._compute_diversity = compute_intra_class_diversity
        self._compute_contrast = compute_contrast_and_saturation

    @torch.no_grad()
    def measure_latency_trials(
        self,
        guidance_scale: float,
        batch_size: int = 16,
        steps: int = 25,
        img_size: int = 28,
        eta: float = 0.0,
        num_warmup: int = 1,
        num_trials: int = 3,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Runs warm-up iterations and repeated latency trials to measure median latency and variability."""
        shape = (batch_size, self.model.in_channels, img_size, img_size)
        y = torch.zeros((batch_size,), dtype=torch.long, device=self.device)

        # 1. Warm-up iterations (excluded from timing)
        for _ in range(num_warmup):
            self.sampler.sample(
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
            torch.cuda.reset_peak_memory_stats(self.device)

        # 2. Repeated timed trials
        trial_times_s: list[float] = []
        for trial_idx in range(num_trials):
            start = time.perf_counter()
            self.sampler.sample(
                shape=shape,
                num_inference_steps=min(steps, self.schedule.timesteps),
                eta=eta,
                y=y,
                guidance_scale=guidance_scale,
                seed=seed + trial_idx + 1,
                device=self.device,
                show_progress=False,
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            trial_times_s.append(time.perf_counter() - start)

        median_latency_s = float(statistics.median(trial_times_s))
        std_latency_s = float(statistics.stdev(trial_times_s)) if len(trial_times_s) > 1 else 0.0

        sorted_times = sorted(trial_times_s)
        # Interquartile range (or simple max-min spread if N < 4)
        if len(sorted_times) >= 4:
            q25 = sorted_times[len(sorted_times) // 4]
            q75 = sorted_times[(len(sorted_times) * 3) // 4]
            iqr_latency_s = q75 - q25
        else:
            iqr_latency_s = sorted_times[-1] - sorted_times[0]

        latency_per_img_ms = (median_latency_s / batch_size) * 1000.0
        throughput_img_s = batch_size / max(1e-6, median_latency_s)
        memory_stats = get_memory_stats(self.device)

        return {
            "median_batch_latency_s": round(median_latency_s, 4),
            "latency_std_s": round(std_latency_s, 4),
            "latency_iqr_s": round(iqr_latency_s, 4),
            "median_latency_per_image_ms": round(latency_per_img_ms, 2),
            "throughput_images_per_s": round(throughput_img_s, 3),
            "max_memory_mb": memory_stats["max_allocated_mb"],
            "trial_times_s": [round(t, 4) for t in trial_times_s],
            "num_trials": num_trials,
            "num_warmup": num_warmup,
        }

    @torch.no_grad()
    def evaluate_scale(
        self,
        guidance_scale: float,
        classes: Sequence[int] = tuple(range(10)),
        num_samples_per_class: int = 16,
        steps: int = 25,
        img_size: int = 28,
        eta: float = 0.0,
        seed: int = 42,
        num_warmup: int = 1,
        num_trials: int = 3,
    ) -> dict[str, Any]:
        """Evaluates generative quality metrics and classifier accuracy across all requested classes."""
        per_class_results: dict[int, dict[str, Any]] = {}
        all_samples_dict: dict[int, torch.Tensor] = {}

        class_accuracies: list[float] = []
        diversities: list[float] = []
        contrasts: list[float] = []
        saturations: list[float] = []
        dynamic_ranges: list[float] = []

        for class_idx in classes:
            shape = (num_samples_per_class, self.model.in_channels, img_size, img_size)
            y = torch.full((num_samples_per_class,), class_idx, dtype=torch.long, device=self.device)

            samples, _ = self.sampler.sample(
                shape=shape,
                num_inference_steps=min(steps, self.schedule.timesteps),
                eta=eta,
                y=y,
                guidance_scale=guidance_scale,
                seed=seed + class_idx * 100,
                device=self.device,
                show_progress=False,
            )
            all_samples_dict[class_idx] = samples

            # Descriptive distributional metrics (not direct perceptual quality)
            div = self._compute_diversity(samples)
            contrast_stats = self._compute_contrast(samples)

            diversities.append(div)
            contrasts.append(contrast_stats["mean_contrast"])
            saturations.append(contrast_stats["saturation_rate"])
            dynamic_ranges.append(contrast_stats["dynamic_range"])

            class_metrics = {
                "class_label": int(class_idx),
                "diversity": div,
                "mean_contrast": contrast_stats["mean_contrast"],
                "saturation_rate": contrast_stats["saturation_rate"],
                "dynamic_range": contrast_stats["dynamic_range"],
            }

            # Independent classifier evaluation
            if self.classifier is not None:
                logits = self.classifier(samples)
                preds = logits.argmax(dim=-1)
                acc = float((preds == y).float().mean().item())
                class_metrics["class_accuracy"] = round(acc, 4)
                class_accuracies.append(acc)

            per_class_results[class_idx] = class_metrics

        # Repeated latency benchmark
        latency_info = self.measure_latency_trials(
            guidance_scale=guidance_scale,
            batch_size=num_samples_per_class,
            steps=steps,
            img_size=img_size,
            eta=eta,
            num_warmup=num_warmup,
            num_trials=num_trials,
            seed=seed,
        )

        aggregated: dict[str, Any] = {
            "mean_diversity": round(float(statistics.mean(diversities)), 6),
            "std_diversity": round(float(statistics.stdev(diversities)), 6) if len(diversities) > 1 else 0.0,
            "mean_contrast": round(float(statistics.mean(contrasts)), 6),
            "std_contrast": round(float(statistics.stdev(contrasts)), 6) if len(contrasts) > 1 else 0.0,
            "mean_saturation_rate": round(float(statistics.mean(saturations)), 6),
            "std_saturation_rate": round(float(statistics.stdev(saturations)), 6) if len(saturations) > 1 else 0.0,
            "mean_dynamic_range": round(float(statistics.mean(dynamic_ranges)), 6),
        }

        if class_accuracies:
            aggregated["mean_class_accuracy"] = round(float(statistics.mean(class_accuracies)), 4)
            aggregated["std_class_accuracy"] = round(float(statistics.stdev(class_accuracies)), 4) if len(class_accuracies) > 1 else 0.0

        metadata = {
            "device": str(self.device),
            "batch_size": num_samples_per_class,
            "inference_steps": steps,
            "precision": "float32",
            "num_classes_evaluated": len(classes),
            "samples_per_class": num_samples_per_class,
        }

        return {
            "guidance_scale": float(guidance_scale),
            "metadata": metadata,
            "latency": latency_info,
            "aggregated": aggregated,
            "per_class": per_class_results,
            "samples": all_samples_dict,
        }

    def run_sweep(
        self,
        scales: Sequence[float] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
        classes: Sequence[int] = tuple(range(10)),
        num_samples_per_class: int = 16,
        steps: int = 25,
        img_size: int = 28,
        output_dir: str | Path = "runs/benchmarks/cfg",
        seed: int = 42,
        num_warmup: int = 1,
        num_trials: int = 3,
        save_grids: bool = True,
    ) -> list[dict[str, Any]]:
        """Run full evaluation sweep across guidance scales supporting all classes."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        results = []

        print(f"\n--- Running CFG Quality Sweep across {len(classes)} classes (Steps: {steps}, Device: {self.device}) ---")

        for s in scales:
            res = self.evaluate_scale(
                guidance_scale=s,
                classes=classes,
                num_samples_per_class=num_samples_per_class,
                steps=steps,
                img_size=img_size,
                seed=seed,
                num_warmup=num_warmup,
                num_trials=num_trials,
            )

            # Save representative grid for class 0 if requested
            if save_grids and 0 in res["samples"]:
                grid = tensor_to_pil_grid(res["samples"][0], nrow=int(num_samples_per_class**0.5))
                grid.save(out_path / f"cfg_scale_{s:.1f}.png")

            clean_res = {k: v for k, v in res.items() if k != "samples"}
            results.append(clean_res)

            acc_str = f"Acc={res['aggregated'].get('mean_class_accuracy', 0.0)*100:.1f}% | " if "mean_class_accuracy" in res["aggregated"] else ""
            print(
                f"  Scale {s:4.1f}: {acc_str}"
                f"Diversity={res['aggregated']['mean_diversity']:.4f} | "
                f"Contrast={res['aggregated']['mean_contrast']:.4f} | "
                f"Saturation={res['aggregated']['mean_saturation_rate']*100:.2f}% | "
                f"Latency={res['latency']['median_latency_per_image_ms']:.1f}ms/img "
                f"(±{res['latency']['latency_std_s']*1000:.1f}ms) | "
                f"Throughput={res['latency']['throughput_images_per_s']:.2f} img/s"
            )

        # Save JSON
        json_path = out_path / "cfg_benchmark_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        # Save summary CSV
        csv_path = out_path / "cfg_benchmark_results.csv"
        flat_rows = []
        for r in results:
            flat_row = {
                "guidance_scale": r["guidance_scale"],
                "mean_class_accuracy": r["aggregated"].get("mean_class_accuracy", ""),
                "std_class_accuracy": r["aggregated"].get("std_class_accuracy", ""),
                "mean_diversity": r["aggregated"]["mean_diversity"],
                "std_diversity": r["aggregated"]["std_diversity"],
                "mean_contrast": r["aggregated"]["mean_contrast"],
                "std_contrast": r["aggregated"]["std_contrast"],
                "mean_saturation_rate": r["aggregated"]["mean_saturation_rate"],
                "std_saturation_rate": r["aggregated"]["std_saturation_rate"],
                "median_batch_latency_s": r["latency"]["median_batch_latency_s"],
                "latency_std_s": r["latency"]["latency_std_s"],
                "median_latency_per_image_ms": r["latency"]["median_latency_per_image_ms"],
                "throughput_images_per_s": r["latency"]["throughput_images_per_s"],
                "device": r["metadata"]["device"],
                "batch_size": r["metadata"]["batch_size"],
                "inference_steps": r["metadata"]["inference_steps"],
                "precision": r["metadata"]["precision"],
            }
            flat_rows.append(flat_row)

        if flat_rows:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=flat_rows[0].keys())
                writer.writeheader()
                writer.writerows(flat_rows)

        print(f"CFG Benchmark results saved to: {json_path} and {csv_path}\n")
        return results


