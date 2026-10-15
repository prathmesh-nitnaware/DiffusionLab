"""Training engine for diffusion models."""

import time
from pathlib import Path
from typing import Any

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from diffusionlab.data.transforms import tensor_to_pil_grid
from diffusionlab.diffusion.forward_process import ForwardDiffusion
from diffusionlab.training.checkpointing import save_checkpoint


class DiffusionTrainer:
    """End-to-end trainer for Denoising Diffusion Probabilistic Models (DDPM)."""

    def __init__(
        self,
        model: nn.Module,
        forward_diff: ForwardDiffusion,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        optimizer: optim.Optimizer | None = None,
        scheduler: Any | None = None,
        config: dict[str, Any] | None = None,
        device: torch.device | str | None = None,
    ):
        self.config = config or {}
        train_cfg = self.config.get("training", self.config)

        # Device selection: support CUDA automatically when available, otherwise CPU
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.forward_diff = forward_diff
        self.forward_diff.schedule.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Optimizer setup
        learning_rate = train_cfg.get("lr", 2e-4)
        weight_decay = train_cfg.get("weight_decay", 1e-4)
        self.optimizer = optimizer or optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = scheduler

        # Training hyperparams
        self.epochs = train_cfg.get("epochs", 10)
        self.max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
        self.log_interval = train_cfg.get("log_interval", 10)
        self.sample_interval = train_cfg.get("sample_interval_epochs", 1)
        self.save_interval = train_cfg.get("save_interval_epochs", 1)
        self.num_sample_images = train_cfg.get("num_sample_images", 16)

        # Output paths
        self.run_dir = Path(train_cfg.get("run_dir", "runs/experiment"))
        self.checkpoint_dir = Path(train_cfg.get("checkpoint_dir", self.run_dir / "checkpoints"))
        self.sample_dir = Path(train_cfg.get("sample_dir", self.run_dir / "samples"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        self.writer = SummaryWriter(log_dir=str(self.run_dir / "logs"))

        # Mixed precision setup (only active on CUDA if enabled in config)
        self.use_amp = bool(train_cfg.get("mixed_precision", False) and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.global_step = 0
        self.start_epoch = 0
        self.best_val_loss = float("inf")

    def train_step(self, x_0: torch.Tensor) -> float:
        """Executes a single genuine training step on a clean image batch."""
        self.model.train()
        x_0 = x_0.to(self.device)
        B = x_0.shape[0]

        # 1. Sample uniform random timesteps for each image in batch
        t = torch.randint(0, self.forward_diff.timesteps, (B,), device=self.device, dtype=torch.long)

        # 2. Sample standard Gaussian noise
        noise = torch.randn_like(x_0)

        # 3. Add noise to clean image using forward diffusion
        x_t = self.forward_diff.q_sample(x_0, t, noise=noise)

        # 4. Predict noise with model
        self.optimizer.zero_grad()
        if self.use_amp:
            with torch.cuda.amp.autocast():
                pred_noise = self.model(x_t, t)
                loss = nn.functional.mse_loss(pred_noise, noise)
            self.scaler.scale(loss).backward()
            if self.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            pred_noise = self.model(x_t, t)
            loss = nn.functional.mse_loss(pred_noise, noise)
            loss.backward()
            if self.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

        self.global_step += 1
        return loss.item()

    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate noise prediction MSE on validation dataset without updating weights."""
        if self.val_loader is None or len(self.val_loader) == 0:
            return 0.0

        self.model.eval()
        total_loss = 0.0
        batches = 0

        for batch in self.val_loader:
            x_0 = batch[0] if isinstance(batch, (list, tuple)) else batch
            x_0 = x_0.to(self.device)
            B = x_0.shape[0]

            t = torch.randint(0, self.forward_diff.timesteps, (B,), device=self.device, dtype=torch.long)
            noise = torch.randn_like(x_0)
            x_t = self.forward_diff.q_sample(x_0, t, noise=noise)

            pred_noise = self.model(x_t, t)
            loss = nn.functional.mse_loss(pred_noise, noise)
            total_loss += loss.item()
            batches += 1

        return total_loss / max(1, batches)

    def generate_and_save_samples(self, epoch: int, num_images: int = 16) -> Path:
        """Fast sample generation during training using reverse diffusion."""
        from diffusionlab.diffusion.samplers import DDPMSampler

        self.model.eval()
        sampler = DDPMSampler(self.model, self.forward_diff.schedule)

        # Determine input shape from first batch
        img_channels = self.model.in_channels
        img_size = self.config.get("img_size", 28)
        shape = (num_images, img_channels, img_size, img_size)

        samples, _ = sampler.sample(
            shape=shape,
            device=self.device,
            show_progress=False,
        )

        pil_grid = tensor_to_pil_grid(samples, nrow=int(num_images**0.5))
        sample_path = self.sample_dir / f"samples_epoch_{epoch:04d}.png"
        pil_grid.save(sample_path)
        return sample_path

    def fit(self, smoke_test: bool = False) -> dict[str, Any]:
        """Execute the full training loop or short smoke test.

        Args:
            smoke_test: If True, executes only 1 batch, saves a checkpoint, and exits.

        Returns:
            Dictionary with training history metrics.
        """
        history = {
            "train_loss": [],
            "val_loss": [],
            "epoch_times": [],
            "total_steps": 0,
        }

        print(
            f"Starting {'SMOKE TEST' if smoke_test else 'TRAINING'} | "
            f"Device: {self.device.type.upper()} | Model Parameters: {self.model.num_parameters():,}"
        )

        start_time = time.time()

        for epoch in range(self.start_epoch, self.epochs):
            epoch_start = time.time()
            epoch_loss = 0.0
            steps_in_epoch = 0

            for batch_idx, batch in enumerate(self.train_loader):
                x_0 = batch[0] if isinstance(batch, (list, tuple)) else batch
                step_loss = self.train_step(x_0)
                epoch_loss += step_loss
                steps_in_epoch += 1

                if self.global_step % self.log_interval == 0 or smoke_test:
                    self.writer.add_scalar("Train/Loss_Step", step_loss, self.global_step)
                    print(
                        f"Epoch [{epoch + 1}/{self.epochs}] Step [{self.global_step}] - "
                        f"Loss: {step_loss:.4f}"
                    )

                if smoke_test:
                    print("Smoke test: verified genuine forward pass, loss calculation, backward pass, optimizer step.")
                    break

            avg_train_loss = epoch_loss / max(1, steps_in_epoch)
            history["train_loss"].append(avg_train_loss)
            self.writer.add_scalar("Train/Loss_Epoch", avg_train_loss, epoch + 1)

            # Validation
            val_loss = self.evaluate()
            history["val_loss"].append(val_loss)
            self.writer.add_scalar("Val/Loss_Epoch", val_loss, epoch + 1)

            epoch_duration = time.time() - epoch_start
            history["epoch_times"].append(epoch_duration)

            print(
                f"Epoch [{epoch + 1}/{self.epochs}] Finished in {epoch_duration:.2f}s | "
                f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()

            # Save checkpoint
            if (epoch + 1) % self.save_interval == 0 or smoke_test or epoch == self.epochs - 1:
                latest_path = self.checkpoint_dir / "latest_model.pt"
                save_checkpoint(
                    latest_path,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    epoch=epoch + 1,
                    global_step=self.global_step,
                    loss=avg_train_loss,
                    config=self.config,
                )

                if val_loss < self.best_val_loss and not smoke_test:
                    self.best_val_loss = val_loss
                    best_path = self.checkpoint_dir / "best_model.pt"
                    save_checkpoint(
                        best_path,
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        scaler=self.scaler,
                        epoch=epoch + 1,
                        global_step=self.global_step,
                        loss=val_loss,
                        config=self.config,
                    )

            # Periodic sample generation (skip if smoke_test unless explicitly desired)
            if not smoke_test and (epoch + 1) % self.sample_interval == 0:
                try:
                    self.generate_and_save_samples(epoch + 1, num_images=self.num_sample_images)
                except Exception as e:
                    print(f"[Warning] Sample generation skipped during training: {e}")

            if smoke_test:
                break

        total_duration = time.time() - start_time
        history["total_steps"] = self.global_step
        history["total_duration"] = total_duration
        self.writer.close()
        return history
