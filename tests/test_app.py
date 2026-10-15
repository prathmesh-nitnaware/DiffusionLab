"""Tests for the Interactive AI Research Studio UI and checkpoint loading."""

import json
from pathlib import Path

import gradio as gr
import pytest
from PIL import Image

from app.app import (
    build_studio_app,
    handle_class_explorer_compare,
    handle_forward_diffusion,
    handle_generate_images,
    handle_load_run_details,
    handle_reverse_trajectory,
    load_evaluation_data,
)
from diffusionlab.inference.loader import (
    find_diffusion_checkpoints,
    is_diffusion_checkpoint,
    load_model_and_schedule_from_checkpoint,
)


class TestCheckpointDiscovery:
    """Verify that checkpoint scanning isolates diffusion models and rejects classifiers."""

    def test_classifier_checkpoint_rejected(self):
        classifier_path = Path("runs/classifier/fashion_mnist_classifier.pt")
        if classifier_path.is_file():
            is_diff = is_diffusion_checkpoint(classifier_path)
            assert not is_diff

    def test_find_diffusion_checkpoints_excludes_classifier(self):
        found = find_diffusion_checkpoints("runs")
        paths = [Path(p).resolve() for p in found]
        classifier_path = Path("runs/classifier/fashion_mnist_classifier.pt").resolve()
        assert classifier_path not in paths

    def test_valid_diffusion_checkpoint_accepted(self):
        ckpt_path = Path("runs/fashion_mnist_mini_project/checkpoints/best_model.pt")
        if ckpt_path.is_file():
            is_diff = is_diffusion_checkpoint(ckpt_path)
            assert is_diff

    def test_load_model_and_schedule(self):
        ckpt_path = Path("runs/fashion_mnist_mini_project/checkpoints/best_model.pt")
        if ckpt_path.is_file():
            model, schedule, metadata = load_model_and_schedule_from_checkpoint(ckpt_path, device="cpu")
            assert hasattr(model, "num_parameters")
            assert model.num_parameters() > 100_000
            assert schedule.timesteps == 200
            assert "epoch" in metadata


class TestUIStudioCallbacks:
    """Verify each interactive studio dashboard section against real components."""

    @pytest.fixture
    def primary_checkpoint(self):
        p = Path("runs/fashion_mnist_mini_project/checkpoints/best_model.pt")
        if not p.is_file():
            pytest.skip("Primary mini-project checkpoint not found")
        return str(p)

    @pytest.fixture
    def label_map(self, primary_checkpoint):
        label = "fashion_mnist_mini_project/best_model.pt (Epoch 3, loss 0.1344)"
        return json.dumps({label: primary_checkpoint}), label

    def test_handle_generate_images_missing_checkpoint(self):
        grid, info = handle_generate_images(
            checkpoint_label="",
            class_choice="0: T-shirt/top",
            sampler_name="DDIM (Fast)",
            num_steps=5,
            eta=0.0,
            batch_size=2,
            seed=42,
            guidance_scale=2.0,
            label_to_path_json="{}",
        )
        assert grid is None
        assert "Setup Required" in info

    def test_handle_generate_images_success(self, primary_checkpoint, label_map):
        mapping_json, label = label_map
        grid, info = handle_generate_images(
            checkpoint_label=label,
            class_choice="0: T-shirt/top",
            sampler_name="DDIM (Fast)",
            num_steps=2,
            eta=0.0,
            batch_size=2,
            seed=42,
            guidance_scale=1.5,
            label_to_path_json=mapping_json,
        )
        assert isinstance(grid, Image.Image)
        assert "Generation Completed" in info
        assert "DDIM" in info
        assert "Model Parameters" in info

    def test_handle_forward_diffusion(self):
        pil_orig, pil_noisy, pil_diff, math_md = handle_forward_diffusion(
            class_choice="0: T-shirt/top",
            timestep=100,
            noise_seed=42,
        )
        assert isinstance(pil_orig, Image.Image)
        assert isinstance(pil_noisy, Image.Image)
        assert isinstance(pil_diff, Image.Image)
        assert "closed-form" in math_md.lower()
        assert "signal-to-noise ratio" in math_md.lower()
        assert "100" in math_md

    def test_handle_reverse_trajectory(self, primary_checkpoint, label_map):
        mapping_json, label = label_map
        strip_img, md_info = handle_reverse_trajectory(
            checkpoint_label=label,
            class_choice="1: Trouser",
            steps=4,
            seed=123,
            label_to_path_json=mapping_json,
        )
        assert isinstance(strip_img, Image.Image)
        assert "Reverse Denoising Trajectory" in md_info
        assert "discrete states" in md_info

    def test_handle_load_run_details(self):
        runs = ["runs/fashion_mnist_mini_project"]
        plot_img, kpi_html, config_json = handle_load_run_details(runs[0])
        assert isinstance(plot_img, Image.Image)
        assert "Completed Epochs" in kpi_html
        assert "Final Train Loss" in kpi_html
        assert "U-Net Parameters" in kpi_html
        parsed = json.loads(config_json)
        assert "base_channels" in parsed or "model" in parsed

    def test_handle_class_explorer_compare(self, primary_checkpoint, label_map):
        mapping_json, label = label_map
        grid, summary = handle_class_explorer_compare(
            checkpoint_label=label,
            selected_classes=["0: T-shirt/top", "1: Trouser"],
            seed=42,
            label_to_path_json=mapping_json,
        )
        assert isinstance(grid, Image.Image)
        assert "Multi-Class Comparative Grid" in summary
        assert "T-shirt/top" in summary

    def test_load_evaluation_data(self):
        kpis, tradeoff_img, dynamics_img, prog_img = load_evaluation_data()
        assert isinstance(tradeoff_img, Image.Image)
        assert isinstance(dynamics_img, Image.Image)
        assert isinstance(prog_img, Image.Image)
        assert "Official Test Set Accuracy" in kpis

    def test_build_studio_app(self):
        demo = build_studio_app()
        assert isinstance(demo, gr.Blocks)
        assert "DiffusionLab" in demo.title
