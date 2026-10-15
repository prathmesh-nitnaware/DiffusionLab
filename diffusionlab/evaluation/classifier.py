"""Independent classifier for evaluating class consistency on Fashion-MNIST."""

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from diffusionlab.data.datasets import get_dataloaders


class FashionMNISTClassifier(nn.Module):
    """Compact convolutional classifier for Fashion-MNIST (1x28x28 -> 10 classes).

    Used as an independent evaluator of class consistency for generated diffusion samples.
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 7x7
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),  # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Expects inputs normalized in range [-1, 1]."""
        feat = self.features(x)
        return self.classifier(feat)


@torch.no_grad()
def evaluate_classifier_accuracy(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device | str = "cpu",
    max_batches: int | None = None,
) -> float:
    """Evaluates top-1 classification accuracy on real held-out data."""
    device = torch.device(device)
    model.eval()
    correct = 0
    total = 0

    for i, batch in enumerate(data_loader):
        if max_batches is not None and i >= max_batches:
            break
        if not (isinstance(batch, (list, tuple)) and len(batch) >= 2):
            continue
        images, labels = batch[0].to(device), batch[1].to(device)
        logits = model(images)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / max(1, total)


@torch.no_grad()
def evaluate_generated_accuracy(
    classifier: nn.Module,
    samples: torch.Tensor,
    target_labels: torch.Tensor,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Evaluates top-1 class accuracy on generated samples.

    Args:
        classifier: Pre-trained Fashion-MNIST classifier.
        samples: Generated images tensor (N, 1, 28, 28) in range [-1, 1].
        target_labels: Intended class labels (N,).
        device: Device to run evaluation on.

    Returns:
        Dict containing:
            - 'overall_accuracy': Fraction of samples where pred == target.
            - 'per_class_accuracy': Dict mapping class_idx to accuracy.
            - 'predictions': List of predicted class integers.
    """
    device = torch.device(device)
    classifier.eval()
    samples = samples.to(device)
    target_labels = target_labels.to(device)

    logits = classifier(samples)
    preds = logits.argmax(dim=-1)

    correct_mask = preds == target_labels
    overall_acc = correct_mask.float().mean().item()

    per_class_acc: dict[int, float] = {}
    for c in torch.unique(target_labels).tolist():
        class_mask = target_labels == c
        if class_mask.sum() > 0:
            per_class_acc[int(c)] = float(correct_mask[class_mask].float().mean().item())

    return {
        "overall_accuracy": round(float(overall_acc), 4),
        "per_class_accuracy": per_class_acc,
        "predictions": preds.cpu().tolist(),
    }


def train_eval_classifier(
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 5,
    lr: float = 0.001,
    device: torch.device | str = "cpu",
    save_path: str | Path | None = None,
) -> tuple[FashionMNISTClassifier, float]:
    """Trains the evaluator classifier on real Fashion-MNIST data."""
    device = torch.device(device)
    model = FashionMNISTClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print(f"Training independent Fashion-MNIST evaluator classifier for {epochs} epochs on {device}...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        batches = 0
        for batch in train_loader:
            if not (isinstance(batch, (list, tuple)) and len(batch) >= 2):
                continue
            images, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batches += 1

        val_acc = evaluate_classifier_accuracy(model, val_loader, device=device)
        print(f"  Classifier Epoch [{epoch + 1}/{epochs}] - Loss: {running_loss/max(1, batches):.4f} | Real Val Acc: {val_acc*100:.2f}%")

    final_val_acc = evaluate_classifier_accuracy(model, val_loader, device=device)

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "real_val_accuracy": final_val_acc,
            },
            path,
        )
        print(f"Saved trained classifier to: {path} (Real Val Accuracy: {final_val_acc*100:.2f}%)")

    return model, final_val_acc


def get_official_test_loader(data_dir: str = "data", batch_size: int = 128) -> DataLoader:
    """Loads the official 10,000-sample Fashion-MNIST test dataset (train=False).

    Guarantees evaluation against the standard benchmark split rather than a validation subset.
    """
    from torchvision.datasets import FashionMNIST

    from diffusionlab.data.transforms import get_diffusion_transforms

    transform = get_diffusion_transforms(28)
    test_ds = FashionMNIST(root=data_dir, train=False, download=False, transform=transform)
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False)


def load_or_train_classifier(
    data_dir: str = "data",
    ckpt_path: str | Path = "runs/classifier/fashion_mnist_classifier.pt",
    device: torch.device | str = "cpu",
) -> tuple[FashionMNISTClassifier, dict[str, Any]]:
    """Loads existing trained classifier or trains a fresh one and returns verified data provenance."""
    device = torch.device(device)
    ckpt_file = Path(ckpt_path)

    # 1. Get real validation dataloader for verification
    train_loader, val_loader = get_dataloaders(
        dataset_name="fashion_mnist",
        data_dir=data_dir,
        img_size=28,
        batch_size=64,
        val_split=0.1,
        num_workers=0,
        use_synthetic_fallback=False,  # Strict: fail rather than silently falling back to synthetic
    )

    if ckpt_file.is_file():
        try:
            ckpt = torch.load(ckpt_file, map_location=device)
            model = FashionMNISTClassifier().to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
        except Exception as e:
            print(f"Could not load existing classifier checkpoint ({e}), retraining...")
            model, _ = train_eval_classifier(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=5,
                device=device,
                save_path=ckpt_file,
            )
    else:
        model, _ = train_eval_classifier(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=5,
            device=device,
            save_path=ckpt_file,
        )

    # Verify both splits rigorously
    val_acc = evaluate_classifier_accuracy(model, val_loader, device=device)
    test_loader = get_official_test_loader(data_dir=data_dir)
    official_test_acc = evaluate_classifier_accuracy(model, test_loader, device=device)

    provenance = {
        "checkpoint_path": str(ckpt_file.resolve()),
        "dataset_name": "FashionMNIST",
        "data_directory": str(Path(data_dir).resolve()),
        "is_synthetic_fallback": False,
        "official_test_accuracy": round(float(official_test_acc), 4),
        "official_test_samples": len(test_loader.dataset),
        "train_val_split_accuracy": round(float(val_acc), 4),
        "train_val_split_samples": len(val_loader.dataset),
        "train_samples": len(train_loader.dataset),
    }

    print(
        f"Verified Classifier: Official Test Acc = {official_test_acc*100:.2f}% (10k images) | "
        f"Val Split Acc = {val_acc*100:.2f}% (6k images)"
    )
    return model, provenance
