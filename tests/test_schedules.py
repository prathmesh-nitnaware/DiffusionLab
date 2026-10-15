"""Tests for noise schedules and diffusion coefficients."""

import pytest
import torch

from diffusionlab.diffusion.schedules import (
    BetaSchedule,
    cosine_beta_schedule,
    extract,
    get_beta_schedule,
    linear_beta_schedule,
)


class TestSchedules:
    """Test suite for beta schedules and coefficient extraction."""

    def test_linear_schedule_shapes_and_bounds(self):
        T = 1000
        betas = linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02)
        assert betas.shape == (T,)
        assert (betas > 0.0).all()
        assert (betas < 1.0).all()
        assert (betas[1:] >= betas[:-1]).all()
        assert pytest.approx(betas[0].item(), rel=1e-3) == 1e-4
        assert pytest.approx(betas[-1].item(), rel=1e-3) == 0.02

    def test_cosine_schedule_shapes_and_bounds(self):
        T = 1000
        betas = cosine_beta_schedule(T, s=0.008, max_beta=0.999)
        assert betas.shape == (T,)
        assert (betas > 0.0).all()
        assert (betas <= 0.999).all()
        assert torch.isfinite(betas).all()

    def test_beta_schedule_coefficients_finite(self):
        for schedule_type in ["linear", "cosine"]:
            sched = BetaSchedule(schedule_type=schedule_type, timesteps=500)
            for attr in [
                "betas",
                "alphas",
                "alphas_cumprod",
                "alphas_cumprod_prev",
                "sqrt_alphas_cumprod",
                "sqrt_one_minus_alphas_cumprod",
                "posterior_variance",
                "posterior_mean_coef1",
                "posterior_mean_coef2",
            ]:
                tensor = getattr(sched, attr)
                assert tensor.shape == (500,), f"{attr} shape mismatch"
                assert torch.isfinite(tensor).all(), f"{attr} contains non-finite values in {schedule_type}"

            # Check that alpha_cumprod is monotonically decreasing
            diffs = sched.alphas_cumprod[1:] - sched.alphas_cumprod[:-1]
            assert (diffs <= 0.0).all(), f"alphas_cumprod not non-increasing in {schedule_type}"
            assert sched.alphas_cumprod[0] > 0.9, "alpha_bar_0 should be close to 1"
            assert sched.alphas_cumprod[-1] < 0.1, "alpha_bar_T should be close to 0"

    def test_extract_function_broadcasting(self):
        T = 100
        a = torch.linspace(1.0, 10.0, T)
        t = torch.tensor([0, 25, 50, 99], dtype=torch.long)
        x_shape = (4, 1, 28, 28)

        extracted = extract(a, t, x_shape)
        assert extracted.shape == (4, 1, 1, 1)
        assert extracted[0, 0, 0, 0] == a[0]
        assert extracted[1, 0, 0, 0] == a[25]
        assert extracted[2, 0, 0, 0] == a[50]
        assert extracted[3, 0, 0, 0] == a[99]

    def test_extract_bounds_checking(self):
        a = torch.linspace(0.1, 0.9, 100)
        x_shape = (2, 3, 32, 32)
        with pytest.raises(IndexError):
            extract(a, torch.tensor([100, 10]), x_shape)
        with pytest.raises(IndexError):
            extract(a, torch.tensor([-1, 10]), x_shape)

    def test_invalid_schedule_parameters(self):
        with pytest.raises(ValueError):
            linear_beta_schedule(0)
        with pytest.raises(ValueError):
            linear_beta_schedule(100, beta_start=-0.01)
        with pytest.raises(ValueError):
            linear_beta_schedule(100, beta_start=0.05, beta_end=0.02)
        with pytest.raises(ValueError):
            cosine_beta_schedule(-10)
        with pytest.raises(ValueError):
            get_beta_schedule("invalid_schedule")
