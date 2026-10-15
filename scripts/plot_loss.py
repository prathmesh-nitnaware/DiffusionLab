import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_loss(history_path: str | Path, output_path: str | Path | None = None) -> Path:
    """Generate loss curve visualization from training_history.json or run directory."""
    path = Path(history_path)
    if path.is_dir():
        path = path / "training_history.json"

    if not path.is_file():
        raise FileNotFoundError(f"History file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    train_loss = data.get("train_loss", [])
    val_loss = data.get("val_loss", [])

    if not train_loss:
        raise ValueError(f"No train_loss entries found in {path}")

    epochs = list(range(1, len(train_loss) + 1))

    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(epochs, train_loss, marker="o", color="#2563EB", linewidth=2, label="Train Loss (MSE)")
    if val_loss and any(v > 0 for v in val_loss):
        plt.plot(epochs, val_loss, marker="s", color="#DC2626", linewidth=2, linestyle="--", label="Val Loss (MSE)")

    plt.title("DiffusionLab: Training & Validation Loss", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Noise Prediction Loss (MSE)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()

    if output_path is None:
        output_path = path.parent / "training_loss.png"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    print(f"Loss plot saved to: {output_path.resolve()}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Plot DiffusionLab training loss curves.")
    parser.add_argument(
        "--history",
        type=str,
        default="runs/fashion_mnist_mini_project/training_history.json",
        help="Path to training_history.json or run directory",
    )
    parser.add_argument("--output", type=str, default=None, help="Destination PNG file path")
    args = parser.parse_args()

    plot_loss(args.history, args.output)


if __name__ == "__main__":
    main()
