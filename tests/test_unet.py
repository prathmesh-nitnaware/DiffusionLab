"""Tests for custom U-Net architecture."""

import torch
from torch import optim

from diffusionlab.models.embeddings import SinusoidalTimeEmbedding, TimestepMLP
from diffusionlab.models.resnet_blocks import AttentionBlock, ResnetBlock
from diffusionlab.models.unet import UNet


class TestUNet:
    """Test suite for UNet and submodules."""

    def test_sinusoidal_embedding_shapes(self):
        dim = 64
        embed = SinusoidalTimeEmbedding(dim=dim)
        t = torch.tensor([0, 10, 500, 999], dtype=torch.long)
        out = embed(t)
        assert out.shape == (4, dim)
        assert torch.isfinite(out).all()

        # Check that different timesteps produce different embeddings
        assert not torch.allclose(out[0], out[1])

    def test_timestep_mlp(self):
        mlp = TimestepMLP(in_features=32, out_features=128)
        t = torch.tensor([0, 100], dtype=torch.long)
        out = mlp(t)
        assert out.shape == (2, 128)
        assert torch.isfinite(out).all()

    def test_resnet_block_forward_and_backward(self):
        block = ResnetBlock(in_channels=16, out_channels=32, time_emb_dim=64)
        x = torch.randn(2, 16, 14, 14, requires_grad=True)
        time_emb = torch.randn(2, 64)
        out = block(x, time_emb)

        assert out.shape == (2, 32, 14, 14)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_attention_block(self):
        attn = AttentionBlock(channels=32, num_heads=4)
        x = torch.randn(2, 32, 7, 7)
        out = attn(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_unet_forward_shape(self):
        model = UNet(
            in_channels=1,
            out_channels=1,
            base_channels=16,
            channel_multipliers=(1, 2),
            num_res_blocks=1,
        )
        x = torch.randn(2, 1, 28, 28)
        t = torch.tensor([10, 500], dtype=torch.long)
        out = model(x, t)

        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_unet_timestep_conditioning_sensitivity(self):
        model = UNet(
            in_channels=1,
            base_channels=16,
            channel_multipliers=(1, 2),
            num_res_blocks=1,
        )
        model.eval()
        x = torch.randn(1, 1, 28, 28)
        t1 = torch.tensor([10], dtype=torch.long)
        t2 = torch.tensor([900], dtype=torch.long)

        with torch.no_grad():
            out1 = model(x, t1)
            out2 = model(x, t2)

        # Conditioning on different timesteps should yield noticeably different outputs
        assert not torch.allclose(out1, out2, atol=1e-3)

    def test_unet_backward_and_optimizer_step(self):
        model = UNet(
            in_channels=1,
            base_channels=16,
            channel_multipliers=(1, 2),
            num_res_blocks=1,
        )
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)

        # Snapshot initial weights
        first_param = next(model.parameters())
        initial_weight = first_param.clone()

        x = torch.randn(2, 1, 28, 28)
        t = torch.tensor([5, 50], dtype=torch.long)
        target = torch.randn_like(x)

        pred = model(x, t)
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()

        # Check that all trainable parameters received finite gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has None grad"
                assert torch.isfinite(param.grad).all(), f"Parameter {name} has non-finite grad"

        optimizer.step()
        assert not torch.allclose(first_param, initial_weight), "Optimizer step did not update parameter weights"

    def test_unet_odd_spatial_resolution(self):
        # 27x27 downsampled: 27 -> 14 -> 7. Upsampled: 7 -> 14 -> 28 vs 27. Should handle mismatch gracefully.
        model = UNet(
            in_channels=1,
            base_channels=16,
            channel_multipliers=(1, 2),
            num_res_blocks=1,
        )
        x = torch.randn(2, 1, 27, 27)
        t = torch.tensor([20, 80], dtype=torch.long)
        out = model(x, t)
        assert out.shape == (2, 1, 27, 27)
