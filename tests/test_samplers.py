"""Tests for DDPM and DDIM samplers."""

import pytest
import torch

from diffusionlab.diffusion.samplers import DDIMSampler, DDPMSampler
from diffusionlab.diffusion.schedules import BetaSchedule
from diffusionlab.models.unet import UNet


class TestSamplers:
    """Test suite for DDPM and DDIM reverse diffusion samplers."""

    @pytest.fixture
    def setup_model_and_schedule(self):
        schedule = BetaSchedule("linear", timesteps=20)
        model = UNet(
            in_channels=1,
            base_channels=8,
            channel_multipliers=(1,),
            num_res_blocks=1,
        )
        return model, schedule

    def test_ddpm_sampling_shape_and_finite(self, setup_model_and_schedule):
        model, schedule = setup_model_and_schedule
        sampler = DDPMSampler(model, schedule)
        shape = (2, 1, 14, 14)

        samples, intermediates = sampler.sample(
            shape=shape,
            device="cpu",
            seed=42,
            save_intermediate_steps=5,
            show_progress=False,
        )

        assert samples.shape == shape
        assert torch.isfinite(samples).all()
        assert len(intermediates) > 1

    def test_ddpm_reproducibility_with_seed(self, setup_model_and_schedule):
        model, schedule = setup_model_and_schedule
        sampler = DDPMSampler(model, schedule)
        shape = (1, 1, 14, 14)

        sample1, _ = sampler.sample(shape=shape, device="cpu", seed=123, show_progress=False)
        sample2, _ = sampler.sample(shape=shape, device="cpu", seed=123, show_progress=False)

        assert torch.allclose(sample1, sample2)

    def test_ddim_sampling_steps_and_deterministic_eta0(self, setup_model_and_schedule):
        model, schedule = setup_model_and_schedule
        sampler = DDIMSampler(model, schedule)
        shape = (2, 1, 14, 14)

        # Sample with 5 inference steps and eta=0.0 (deterministic ODE)
        sample1, intermediates = sampler.sample(
            shape=shape,
            num_inference_steps=5,
            eta=0.0,
            device="cpu",
            seed=999,
            save_intermediate=True,
            show_progress=False,
        )

        assert sample1.shape == shape
        assert torch.isfinite(sample1).all()
        assert len(intermediates) == 6  # initial x_T + 5 steps

        sample2, _ = sampler.sample(
            shape=shape,
            num_inference_steps=5,
            eta=0.0,
            device="cpu",
            seed=999,
            show_progress=False,
        )
        assert torch.allclose(sample1, sample2)

    def test_ddim_invalid_inference_steps(self, setup_model_and_schedule):
        model, schedule = setup_model_and_schedule
        sampler = DDIMSampler(model, schedule)
        with pytest.raises(ValueError):
            sampler.sample(shape=(1, 1, 14, 14), num_inference_steps=100)  # schedule only has 20 timesteps
