"""Comprehensive tests for Classifier-Free Guidance (CFG).

Covers:
- Label validation and null label configuration
- Condition dropout logic
- Conditional forward pass and batched forward_with_cfg mathematical correctness
- DDPM and DDIM conditional sampling with guidance scale
- Backward compatibility with unconditional models and checkpoints
"""

import pytest
import torch
from torch import nn

from diffusionlab.diffusion.samplers import DDIMSampler, DDPMSampler
from diffusionlab.diffusion.schedules import get_beta_schedule
from diffusionlab.inference.loader import load_model_and_schedule_from_checkpoint
from diffusionlab.models.unet import UNet
from diffusionlab.training.checkpointing import save_checkpoint


def test_unet_unconditional_backward_compatibility():
    """Verify UNet initialized with num_classes=None preserves unconditional behavior."""
    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=16,
        channel_multipliers=(1, 2),
        num_res_blocks=1,
        num_classes=None,
    )
    assert model.num_classes is None
    assert model.null_label is None
    assert model.label_emb is None

    x = torch.randn(2, 1, 16, 16)
    t = torch.tensor([5, 10])

    # Standard forward pass
    out = model(x, t)
    assert out.shape == (2, 1, 16, 16)

    # forward_with_cfg should fall back gracefully
    out_cfg = model.forward_with_cfg(x, t, y=None, guidance_scale=2.0)
    assert torch.allclose(out, out_cfg)


def test_unet_conditional_initialization_and_label_validation():
    """Verify conditional UNet configuration and strict label validation."""
    num_classes = 10
    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=16,
        channel_multipliers=(1, 2),
        num_res_blocks=1,
        num_classes=num_classes,
    )

    assert model.num_classes == 10
    assert model.null_label == 10
    assert isinstance(model.label_emb, nn.Embedding)
    assert model.label_emb.num_embeddings == 11

    x = torch.randn(4, 1, 16, 16)
    t = torch.tensor([1, 2, 3, 4])

    # Valid labels: 0..9 and null_label (10)
    valid_y = torch.tensor([0, 5, 9, 10])
    out = model(x, t, y=valid_y)
    assert out.shape == (4, 1, 16, 16)

    # Out-of-bounds label (too high)
    with pytest.raises(ValueError, match="Class labels must be in range"):
        model(x, t, y=torch.tensor([0, 1, 11, 2]))

    # Negative label
    with pytest.raises(ValueError, match="Class labels must be in range"):
        model(x, t, y=torch.tensor([-1, 0, 1, 2]))


def test_condition_dropout_logic():
    """Verify condition dropout maps labels to null_label at expected probability."""
    torch.manual_seed(42)
    p_uncond = 0.2
    null_label = 10
    y = torch.randint(0, 10, (10000,))

    mask = torch.rand(y.shape) < p_uncond
    y_dropped = torch.where(mask, torch.full_like(y, null_label), y)

    dropout_rate = (y_dropped == null_label).float().mean().item()
    # Should be around 0.2 within 3 standard deviations
    assert 0.18 <= dropout_rate <= 0.22


def test_batched_forward_with_cfg_mathematics():
    """Verify that batched forward_with_cfg exactly satisfies the CFG formula:

    eps_guided = eps_uncond + s * (eps_cond - eps_uncond)
    """
    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=16,
        channel_multipliers=(1, 2),
        num_res_blocks=1,
        num_classes=10,
    )
    model.eval()

    x = torch.randn(4, 1, 16, 16)
    t = torch.tensor([10, 20, 30, 40])
    y = torch.tensor([0, 3, 7, 9])
    guidance_scale = 3.0

    # 1. Batched forward_with_cfg
    with torch.no_grad():
        eps_guided = model.forward_with_cfg(x, t, y=y, guidance_scale=guidance_scale)

        # 2. Independent passes for verification
        eps_cond = model(x, t, y=y)
        null_y = torch.full_like(y, model.null_label)
        eps_uncond = model(x, t, y=null_y)

        # Mathematical CFG formula
        expected = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

    assert torch.allclose(eps_guided, expected, atol=1e-5), "Batched CFG does not match formula!"

    # Test scale = 1.0 matches pure conditional
    with torch.no_grad():
        eps_s1 = model.forward_with_cfg(x, t, y=y, guidance_scale=1.0)
        assert torch.allclose(eps_s1, eps_cond, atol=1e-5)

    # Test scale = 0.0 matches pure unconditional
    with torch.no_grad():
        eps_s0 = model.forward_with_cfg(x, t, y=y, guidance_scale=0.0)
        assert torch.allclose(eps_s0, eps_uncond, atol=1e-5)


def test_samplers_with_cfg():
    """Verify DDPM and DDIM sampling with classifier-free guidance."""
    schedule = get_beta_schedule("linear", timesteps=10, beta_start=1e-4, beta_end=0.02)
    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=8,
        channel_multipliers=(1, 2),
        num_res_blocks=1,
        num_classes=10,
    )
    model.eval()

    shape = (2, 1, 8, 8)
    y = torch.tensor([1, 8])

    # 1. DDPM Sampler with CFG
    ddpm_sampler = DDPMSampler(model, schedule)
    samples_ddpm, _ = ddpm_sampler.sample(
        shape=shape,
        y=y,
        guidance_scale=2.5,
        device="cpu",
        show_progress=False,
    )
    assert samples_ddpm.shape == shape
    assert not torch.isnan(samples_ddpm).any()

    # 2. DDIM Sampler with CFG
    ddim_sampler = DDIMSampler(model, schedule)
    samples_ddim, _ = ddim_sampler.sample(
        shape=shape,
        num_inference_steps=5,
        y=y,
        guidance_scale=2.5,
        device="cpu",
        show_progress=False,
    )
    assert samples_ddim.shape == shape
    assert not torch.isnan(samples_ddim).any()


def test_checkpoint_compatibility_loader(tmp_path):
    """Verify checkpoint saving and loading maintains compatibility for both conditional and unconditional models."""
    # 1. Unconditional model checkpoint
    uncond_model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=16,
        channel_multipliers=(1, 2),
        num_res_blocks=1,
        num_classes=None,
    )
    uncond_path = tmp_path / "uncond.pt"
    save_checkpoint(
        checkpoint_path=uncond_path,
        model=uncond_model,
        epoch=1,
        global_step=10,
        config={"model": {"base_channels": 16, "channel_multipliers": [1, 2]}},
    )

    loaded_uncond, _, _ = load_model_and_schedule_from_checkpoint(uncond_path, device="cpu")
    assert loaded_uncond.num_classes is None
    assert loaded_uncond.label_emb is None

    # 2. Conditional model checkpoint
    cond_model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=16,
        channel_multipliers=(1, 2),
        num_res_blocks=1,
        num_classes=10,
    )
    cond_path = tmp_path / "cond.pt"
    save_checkpoint(
        checkpoint_path=cond_path,
        model=cond_model,
        epoch=1,
        global_step=10,
        config={"model": {"base_channels": 16, "channel_multipliers": [1, 2], "num_classes": 10}},
    )

    loaded_cond, _, _ = load_model_and_schedule_from_checkpoint(cond_path, device="cpu")
    assert loaded_cond.num_classes == 10
    assert loaded_cond.null_label == 10
    assert loaded_cond.label_emb is not None
