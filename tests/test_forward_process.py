"""Tests for forward diffusion process."""

import pytest
import torch

from diffusionlab.diffusion.forward_process import ForwardDiffusion
from diffusionlab.diffusion.schedules import BetaSchedule


class TestForwardDiffusion:
    """Test suite for ForwardDiffusion."""

    @pytest.fixture
    def forward_diff(self):
        sched = BetaSchedule(schedule_type="linear", timesteps=1000)
        return ForwardDiffusion(sched)

    def test_output_shape(self, forward_diff):
        batch_size = 4
        x_0 = torch.randn(batch_size, 1, 28, 28)
        t = torch.tensor([0, 100, 500, 999], dtype=torch.long)
        x_t = forward_diff.q_sample(x_0, t)

        assert x_t.shape == x_0.shape
        assert torch.isfinite(x_t).all()

    def test_reproducibility_with_explicit_noise(self, forward_diff):
        batch_size = 2
        x_0 = torch.randn(batch_size, 1, 28, 28)
        t = torch.tensor([50, 200], dtype=torch.long)
        noise = torch.randn_like(x_0)

        x_t_1 = forward_diff.q_sample(x_0, t, noise=noise)
        x_t_2 = forward_diff.q_sample(x_0, t, noise=noise)

        assert torch.allclose(x_t_1, x_t_2)

    def test_reconstruction_from_noise(self, forward_diff):
        x_0 = torch.randn(4, 1, 28, 28)
        t = torch.tensor([10, 50, 100, 500], dtype=torch.long)
        noise = torch.randn_like(x_0)

        x_t = forward_diff.q_sample(x_0, t, noise=noise)
        x_0_pred = forward_diff.predict_start_from_noise(x_t, t, noise)

        assert torch.allclose(x_0, x_0_pred, atol=1e-5)

    def test_boundary_behavior(self, forward_diff):
        # At t=0, alpha_bar is very close to 1, so x_t should be very close to x_0
        x_0 = torch.ones(1, 1, 14, 14) * 2.0
        t_0 = torch.tensor([0], dtype=torch.long)
        noise = torch.zeros_like(x_0)
        x_t0 = forward_diff.q_sample(x_0, t_0, noise=noise)
        assert torch.allclose(x_t0, x_0, atol=1e-2)

        # At t=T-1, alpha_bar is very small, so signal should be mostly noise
        t_last = torch.tensor([999], dtype=torch.long)
        noise = torch.ones_like(x_0) * 5.0
        x_t_last = forward_diff.q_sample(x_0, t_last, noise=noise)
        # alpha_bar_999 is ~ 1e-4, sqrt(alpha_bar) is ~ 0.01, sqrt(1-alpha_bar) is ~ 0.999
        assert torch.allclose(x_t_last, noise, atol=0.1)

    def test_sample_trajectory(self, forward_diff):
        x_0 = torch.randn(1, 1, 28, 28)
        timesteps_to_view = [0, 100, 500, 999]
        trajectory = forward_diff.sample_trajectory(x_0, timesteps_to_view)

        assert len(trajectory) == 4
        for step_t, x_t in trajectory:
            assert step_t in timesteps_to_view
            assert x_t.shape == (1, 1, 28, 28)
            assert torch.isfinite(x_t).all()

    def test_invalid_inputs(self, forward_diff):
        x_0 = torch.randn(2, 1, 28, 28)
        # Mismatched noise shape
        with pytest.raises(ValueError):
            forward_diff.q_sample(x_0, torch.tensor([0, 1]), noise=torch.randn(3, 1, 28, 28))
        # Mismatched batch size in t
        with pytest.raises(ValueError):
            forward_diff.q_sample(x_0, torch.tensor([0]))
        # Invalid trajectory timestep
        with pytest.raises(ValueError):
            forward_diff.sample_trajectory(x_0, [1500])
