"""Tests for diffusion training engine."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from diffusionlab.diffusion.forward_process import ForwardDiffusion
from diffusionlab.diffusion.schedules import BetaSchedule
from diffusionlab.models.unet import UNet
from diffusionlab.training.trainer import DiffusionTrainer


class TestTraining:
    """Test suite for DiffusionTrainer."""

    @pytest.fixture
    def setup_trainer(self, tmp_path):
        model = UNet(
            in_channels=1,
            base_channels=16,
            channel_multipliers=(1, 2),
            num_res_blocks=1,
        )
        schedule = BetaSchedule("linear", timesteps=100)
        forward_diff = ForwardDiffusion(schedule)

        # In-memory toy dataset
        data = torch.randn(16, 1, 28, 28).clamp(-1.0, 1.0)
        train_loader = DataLoader(TensorDataset(data), batch_size=4, shuffle=True)
        val_loader = DataLoader(TensorDataset(data), batch_size=4, shuffle=False)

        config = {
            "epochs": 1,
            "lr": 1e-3,
            "max_grad_norm": 1.0,
            "run_dir": str(tmp_path / "run"),
            "img_size": 28,
        }

        trainer = DiffusionTrainer(
            model=model,
            forward_diff=forward_diff,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device="cpu",
        )
        return trainer

    def test_single_train_step(self, setup_trainer):
        trainer = setup_trainer
        batch = torch.randn(4, 1, 28, 28)
        loss = trainer.train_step(batch)

        assert isinstance(loss, float)
        assert loss > 0.0
        assert torch.isfinite(torch.tensor(loss))
        assert trainer.global_step == 1

    def test_validation_does_not_update_weights(self, setup_trainer):
        trainer = setup_trainer
        initial_params = [p.clone() for p in trainer.model.parameters()]

        val_loss = trainer.evaluate()
        assert isinstance(val_loss, float)
        assert val_loss > 0.0

        for p_before, p_after in zip(initial_params, trainer.model.parameters()):
            assert torch.equal(p_before, p_after), "Validation modified model parameters!"

    def test_smoke_test_run(self, setup_trainer):
        trainer = setup_trainer
        history = trainer.fit(smoke_test=True)

        assert history["total_steps"] == 1
        assert len(history["train_loss"]) == 1
        assert (trainer.checkpoint_dir / "latest_model.pt").is_file()
        assert (trainer.run_dir / "run_metadata.json").is_file()
        assert (trainer.run_dir / "training_history.json").is_file()

    def test_resume_from_checkpoint_preserves_state(self, setup_trainer, tmp_path):
        from diffusionlab.training.checkpointing import save_checkpoint

        trainer = setup_trainer
        ckpt_path = tmp_path / "test_ckpt.pt"
        save_checkpoint(
            ckpt_path,
            model=trainer.model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            epoch=5,
            global_step=120,
            loss=0.4521,
        )

        # Create a fresh trainer and resume
        fresh_trainer = DiffusionTrainer(
            model=trainer.model,
            forward_diff=trainer.forward_diff,
            train_loader=trainer.train_loader,
            val_loader=trainer.val_loader,
            config=trainer.config,
            device="cpu",
        )
        assert fresh_trainer.start_epoch == 0
        assert fresh_trainer.global_step == 0

        fresh_trainer.resume_from_checkpoint(ckpt_path)
        assert fresh_trainer.start_epoch == 5
        assert fresh_trainer.global_step == 120
        assert pytest.approx(fresh_trainer.best_val_loss, rel=1e-4) == 0.4521

    def test_conditional_train_step_with_dropout(self, tmp_path):
        model = UNet(
            in_channels=1,
            base_channels=16,
            channel_multipliers=(1, 2),
            num_res_blocks=1,
            num_classes=10,
        )
        schedule = BetaSchedule("linear", timesteps=50)
        forward_diff = ForwardDiffusion(schedule)
        data = torch.randn(8, 1, 28, 28)
        labels = torch.randint(0, 10, (8,))
        train_loader = DataLoader(TensorDataset(data, labels), batch_size=4)

        trainer = DiffusionTrainer(
            model=model,
            forward_diff=forward_diff,
            train_loader=train_loader,
            config={"training": {"epochs": 1, "run_dir": str(tmp_path / "cond_run")}},
            device="cpu",
            p_uncond=0.2,
        )

        loss = trainer.train_step(data[:4], y=labels[:4])
        assert isinstance(loss, float)
        assert torch.isfinite(torch.tensor(loss))
        assert trainer.global_step == 1

    def test_synthetic_fallback_disabled_raises_on_invalid_dir(self, monkeypatch):
        import diffusionlab.data.datasets as ds_mod
        from diffusionlab.data.datasets import get_dataloaders

        def mock_download_failure(*args, **kwargs):
            raise RuntimeError("Mock download failure")

        monkeypatch.setattr(ds_mod, "FashionMNIST", mock_download_failure)

        with pytest.raises(RuntimeError, match="Mock download failure"):
            get_dataloaders(
                dataset_name="fashion_mnist",
                use_synthetic_fallback=False,
            )

