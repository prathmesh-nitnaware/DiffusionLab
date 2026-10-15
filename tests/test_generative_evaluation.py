"""Unit tests for generative quality evaluation metrics, independent classifier, and CFG benchmarking suite."""

import torch

from diffusionlab.diffusion.forward_process import ForwardDiffusion
from diffusionlab.diffusion.schedules import get_beta_schedule
from diffusionlab.evaluation.benchmarks import CFGQualityBenchmark
from diffusionlab.evaluation.classifier import (
    FashionMNISTClassifier,
    evaluate_generated_accuracy,
)
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


def test_fashion_mnist_classifier_and_generated_accuracy():
    """Verify independent classifier forward pass and generated accuracy metrics."""
    classifier = FashionMNISTClassifier(num_classes=10)
    classifier.eval()

    # Forward pass shape
    dummy_input = torch.randn(4, 1, 28, 28)
    logits = classifier(dummy_input)
    assert logits.shape == (4, 10)

    # Generated accuracy function
    targets = torch.tensor([0, 1, 2, 3])
    results = evaluate_generated_accuracy(classifier, dummy_input, targets, device="cpu")
    assert "overall_accuracy" in results
    assert "per_class_accuracy" in results
    assert "predictions" in results
    assert 0.0 <= results["overall_accuracy"] <= 1.0


def test_measure_latency_trials():
    """Verify repeated latency trials with warm-up return median and variability."""
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
    latency_info = benchmark.measure_latency_trials(
        guidance_scale=1.0,
        batch_size=2,
        steps=2,
        img_size=8,
        num_warmup=1,
        num_trials=2,
    )

    assert "median_batch_latency_s" in latency_info
    assert "latency_std_s" in latency_info
    assert "latency_iqr_s" in latency_info
    assert "median_latency_per_image_ms" in latency_info
    assert "throughput_images_per_s" in latency_info
    assert len(latency_info["trial_times_s"]) == 2


def test_cfg_quality_benchmark_sweep(tmp_path):
    """Verify CFGQualityBenchmark executes multi-class sweep with classifier and produces structured outputs."""
    schedule = get_beta_schedule("linear", timesteps=10)
    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=8,
        channel_multipliers=(1, 2),
        num_res_blocks=1,
        num_classes=10,
    )
    classifier = FashionMNISTClassifier(num_classes=10)

    benchmark = CFGQualityBenchmark(model, schedule, classifier=classifier, device="cpu")
    results = benchmark.run_sweep(
        scales=[0.0, 2.0],
        classes=(0, 1),
        num_samples_per_class=2,
        steps=2,
        img_size=8,
        output_dir=tmp_path,
        num_warmup=1,
        num_trials=2,
        save_grids=True,
    )

    assert len(results) == 2
    for r in results:
        assert "guidance_scale" in r
        assert "aggregated" in r
        assert "mean_diversity" in r["aggregated"]
        assert "mean_class_accuracy" in r["aggregated"]
        assert "mean_contrast" in r["aggregated"]
        assert "mean_saturation_rate" in r["aggregated"]
        assert "latency" in r
        assert "median_batch_latency_s" in r["latency"]
        assert "metadata" in r
        assert r["metadata"]["num_classes_evaluated"] == 2
        assert "per_class" in r
        assert 0 in r["per_class"]
        assert 1 in r["per_class"]

    assert (tmp_path / "cfg_benchmark_results.json").exists()
    assert (tmp_path / "cfg_benchmark_results.csv").exists()
    assert (tmp_path / "cfg_scale_0.0.png").exists()
    assert (tmp_path / "cfg_scale_2.0.png").exists()
