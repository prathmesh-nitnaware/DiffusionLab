"""Tests for checkpointing, saving, and loading."""

import pytest
import torch
from torch import optim

from diffusionlab.models.unet import UNet
from diffusionlab.training.checkpointing import (
    find_best_checkpoint,
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)


class TestCheckpointing:
    """Test suite for checkpoint saving and loading."""

    def test_save_and_load_roundtrip(self, tmp_path):
        ckpt_path = tmp_path / "model.pt"
        model = UNet(in_channels=1, base_channels=16, channel_multipliers=(1, 2), num_res_blocks=1)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)

        # Do a mock step to change optimizer and scheduler state
        x = torch.randn(2, 1, 14, 14)
        t = torch.tensor([0, 1])
        loss = model(x, t).sum()
        loss.backward()
        optimizer.step()
        scheduler.step()

        save_checkpoint(
            ckpt_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=2,
            global_step=15,
            loss=0.42,
            config={"base_channels": 16},
        )

        assert ckpt_path.is_file()

        # Create new model and restore
        new_model = UNet(in_channels=1, base_channels=16, channel_multipliers=(1, 2), num_res_blocks=1)
        new_optimizer = optim.AdamW(new_model.parameters(), lr=1e-3)
        new_scheduler = optim.lr_scheduler.StepLR(new_optimizer, step_size=1)

        ckpt_data = load_checkpoint(
            ckpt_path,
            model=new_model,
            optimizer=new_optimizer,
            scheduler=new_scheduler,
        )

        assert ckpt_data["epoch"] == 2
        assert ckpt_data["global_step"] == 15
        assert pytest.approx(ckpt_data["loss"]) == 0.42

        # Check parameter equality
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)

    def test_incompatible_model_architecture_error(self, tmp_path):
        ckpt_path = tmp_path / "small_model.pt"
        small_model = UNet(in_channels=1, base_channels=16, channel_multipliers=(1, 2), num_res_blocks=1)
        save_checkpoint(ckpt_path, model=small_model)

        # Larger model architecture
        big_model = UNet(in_channels=1, base_channels=32, channel_multipliers=(1, 2), num_res_blocks=1)

        with pytest.raises(RuntimeError):
            load_checkpoint(ckpt_path, model=big_model)

    def test_find_latest_and_best_checkpoint(self, tmp_path):
        ckpt_dir = tmp_path / "ckpts"
        ckpt_dir.mkdir()

        # No checkpoints initially
        assert find_latest_checkpoint(ckpt_dir) is None
        assert find_best_checkpoint(ckpt_dir) is None

        # Create dummy checkpoints
        model = UNet(in_channels=1, base_channels=8, channel_multipliers=(1,))
        save_checkpoint(ckpt_dir / "ckpt_epoch_1.pt", model=model, epoch=1)
        save_checkpoint(ckpt_dir / "best_model.pt", model=model, epoch=1, loss=0.1)

        assert find_latest_checkpoint(ckpt_dir) is not None
        assert find_best_checkpoint(ckpt_dir).name == "best_model.pt"
