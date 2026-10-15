"""Unit tests for generative quality evaluation metrics and CFG benchmarking suite."""

import torch

from diffusionlab.diffusion.forward_process import ForwardDiffusion
from diffusionlab.diffusion.schedules import get_beta_schedule
from diffusionlab.evaluation.benchmarks import CFGQualityBenchmark
from diffusionlab.evaluation.metrics import (
    compute_contrast_and_saturation,
    compute_intra_class_diversity,
    evaluate_guidance_vector_dynamics,
)
from diffusionlab.models.unet import UNet


def test_intra_class_diversity_bounds():
    """Verify intra-class diversity metric behaves predictably on edge cases."""
    # 1. Single sample should return 0.0
    single = torch.randn(1, 1, 16, 16)
    assert compute_intra_class_diversity(single) == 0.0

    # 2. Identical duplicate samples (complete mode collapse) should yield 0.0 diversity
    base = torch.randn(1, 1, 16, 16)
    identical = base.repeat(4, 1, 1, 1)
    assert compute_intra_class_diversity(identical) == 0.0

    # 3. Independent random noise should yield strictly positive diversity
    random_samples = torch.randn(8, 1, 16, 16)
    div = compute_intra_class_diversity(random_samples)
    assert div > 0.0


def test_contrast_and_saturation_metrics():
    """Verify spatial contrast and saturation metrics on synthetic extreme cases."""
    # 1. Uniform flat samples
    flat = torch.zeros(4, 1, 16, 16)
    stats_flat = compute_contrast_and_saturation(flat)
    assert stats_flat["mean_contrast"] == 0.0
    assert stats_flat["saturation_rate"] == 0.0
    assert stats_flat["dynamic_range"] == 0.0

    # 2. Fully saturated bipolar samples (+1 and -1 checkerboard)
    saturated = torch.ones(4, 1, 16, 16)
    stats_sat = compute_contrast_and_saturation(saturated)
    assert stats_sat["saturation_rate"] == 1.0


def test_guidance_vector_dynamics():
    """Verify evaluation of steering vector magnitude across diffusion timesteps."""
    schedule = get_beta_schedule("linear", timesteps=20)
    forward_diff = ForwardDiffusion(schedule)
    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=8,
        channel_multipliers=(1, 2),
        num_res_blocks=1,
        num_classes=10,
    )

    # Synthetic loader: list of batches (x_0, y)
    x_fake = torch.randn(4, 1, 16, 16)
    y_fake = torch.tensor([0, 1, 2, 3])
    data_loader = [(x_fake, y_fake)]

    timesteps = [5, 10, 15]
    dynamics = evaluate_guidance_vector_dynamics(
        model=model,
        forward_diff=forward_diff,
        data_loader=data_loader,
        device="cpu",
        timesteps=timesteps,
        max_batches=1,
    )

    assert set(dynamics.keys()) == set(timesteps)
    for t_val in timesteps:
        assert isinstance(dynamics[t_val], float)
        assert dynamics[t_val] >= 0.0


def test_cfg_quality_benchmark_sweep(tmp_path):
    """Verify CFGQualityBenchmark executes sweep and produces structured outputs."""
    schedule = get_beta_schedule("linear", timesteps=10)
    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=8,
        channel_multipliers=(1, 2),
        num_res_blocks=1,
        num_classes=10,
    )

    benchmark = CFGQualityBenchmark(model, schedule, device="cpu")
    results = benchmark.run_sweep(
        scales=[0.0, 2.0],
        class_label=1,
        num_samples=2,
        steps=2,
        img_size=8,
        output_dir=tmp_path,
        save_grids=True,
    )

    assert len(results) == 2
    for r in results:
        assert "diversity" in r
        assert "mean_contrast" in r
        assert "saturation_rate" in r
        assert "throughput_images_per_s" in r

    assert (tmp_path / "cfg_benchmark_results.json").exists()
    assert (tmp_path / "cfg_benchmark_results.csv").exists()
    assert (tmp_path / "cfg_scale_0.0.png").exists()
    assert (tmp_path / "cfg_scale_2.0.png").exists()
