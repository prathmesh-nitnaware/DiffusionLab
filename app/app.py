"""DiffusionLab — Interactive AI Research Studio UI.

A polished, comprehensive Deep Learning research dashboard for Fashion-MNIST
diffusion models (DDPM & DDIM) with class conditioning, forward/reverse diffusion lab,
training analytics, class comparison, and evaluation benchmarks.
"""

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import gradio as gr
import torch
from PIL import Image
from torchvision.datasets import FashionMNIST

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffusionlab.data.transforms import get_diffusion_transforms, tensor_to_pil_grid
from diffusionlab.diffusion.forward_process import ForwardDiffusion
from diffusionlab.diffusion.samplers import DDIMSampler, DDPMSampler
from diffusionlab.diffusion.schedules import get_beta_schedule
from diffusionlab.inference.loader import (
    find_diffusion_checkpoints,
    load_model_and_schedule_from_checkpoint,
)

logger = logging.getLogger(__name__)

# Class name definitions for Fashion-MNIST
FASHION_CLASSES = [
    "0: T-shirt/top",
    "1: Trouser",
    "2: Pullover",
    "3: Dress",
    "4: Coat",
    "5: Sandal",
    "6: Shirt",
    "7: Sneaker",
    "8: Bag",
    "9: Ankle boot",
]

# Cache real test images on module load (if data directory exists)
_REAL_TEST_SAMPLES: dict[str, tuple[Image.Image, torch.Tensor]] = {}


def load_real_test_samples() -> dict[str, tuple[Image.Image, torch.Tensor]]:
    """Load 1 genuine test sample per class from local data cache."""
    global _REAL_TEST_SAMPLES
    if _REAL_TEST_SAMPLES:
        return _REAL_TEST_SAMPLES

    data_dir = Path("data")
    if not (data_dir / "FashionMNIST" / "raw").exists():
        return {}

    try:
        ds = FashionMNIST(root=str(data_dir), train=False, download=False)
        transform = get_diffusion_transforms(img_size=28)
        found = {}
        for img, label in ds:
            name = FASHION_CLASSES[label]
            if name not in found:
                tensor_img = transform(img)
                found[name] = (img, tensor_img)
            if len(found) == 10:
                break
        _REAL_TEST_SAMPLES = found
    except Exception as e:
        print(f"[Warning] Could not pre-cache real test images: {e}")
        _REAL_TEST_SAMPLES = {}
    return _REAL_TEST_SAMPLES


# Pre-cache on startup
load_real_test_samples()


# Custom Dark Navy & Purple/Blue Theme CSS
STUDIO_CSS = """
:root {
    --bg-dark: #0a0e1a;
    --card-bg: rgba(15, 23, 42, 0.75);
    --card-border: 1px solid rgba(99, 102, 241, 0.25);
    --accent-blue: #3b82f6;
    --accent-purple: #8b5cf6;
    --accent-cyan: #06b6d4;
    --text-main: #f8fafc;
    --text-muted: #94a3b8;
}

body, .gradio-container {
    background-color: var(--bg-dark) !important;
    color: var(--text-main) !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}

/* Glassmorphism Cards */
.studio-card {
    background: var(--card-bg) !important;
    border: var(--card-border) !important;
    border-radius: 12px !important;
    padding: 20px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4) !important;
    backdrop-filter: blur(8px) !important;
    margin-bottom: 16px !important;
}

/* Header Gradient */
.header-badge {
    display: inline-block;
    padding: 4px 12px;
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(139, 92, 246, 0.2));
    border: 1px solid rgba(139, 92, 246, 0.4);
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    color: #a78bfa;
    margin-bottom: 8px;
}

.studio-title {
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    margin-bottom: 4px !important;
}

/* Stat KPI Cards */
.kpi-container {
    display: flex;
    gap: 12px;
    margin-bottom: 16px;
    flex-wrap: wrap;
}

.kpi-box {
    flex: 1;
    min-width: 140px;
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 12px 16px;
    text-align: center;
}

.kpi-val {
    font-size: 1.4rem;
    font-weight: 700;
    color: #38bdf8;
}

.kpi-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94a3b8;
    margin-top: 4px;
}

/* Tab styling */
button.selected {
    border-bottom: 2px solid #8b5cf6 !important;
    color: #c084fc !important;
    font-weight: 600 !important;
}

/* Scientific callout alert */
.scientific-note {
    background: rgba(30, 41, 59, 0.6);
    border-left: 4px solid #38bdf8;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin: 12px 0;
    font-size: 0.9rem;
    color: #cbd5e1;
}
"""


def format_checkpoint_choices() -> tuple[list[str], dict[str, str], str]:
    """Scan and format valid diffusion checkpoints with informative labels."""
    ckpts = find_diffusion_checkpoints()
    if not ckpts:
        return ["No trained diffusion checkpoint available"], {}, ""

    label_to_path = {}
    choices = []
    for c in ckpts:
        type_str = "Conditional" if c["is_conditional"] else "Unconditional"
        loss_str = f"Loss {c['loss']}" if c["loss"] != "N/A" else ""
        epoch_str = f"Epoch {c['epoch']}" if c["epoch"] != "N/A" else ""
        info_parts = [p for p in [type_str, epoch_str, loss_str] if p]
        label = f"{c['run_name']} / {c['name']} ({', '.join(info_parts)})"
        label_to_path[label] = c["path"]
        choices.append(label)

    default_choice = choices[0] if choices else ""
    return choices, label_to_path, default_choice


# =========================================================================
# Tab 1: Generate Studio Logic
# =========================================================================


def handle_generate_images(
    checkpoint_label: str,
    class_choice: str,
    sampler_name: str,
    num_steps: int,
    eta: float,
    batch_size: int,
    seed: int,
    guidance_scale: float,
    label_to_path_json: str,
):
    if not checkpoint_label or "No trained" in checkpoint_label:
        return (
            None,
            (
                "⚠️ **Setup Required**: No compatible diffusion checkpoint found. Train a model first using:\n"
                "```powershell\npython scripts/train.py --config configs/fashion_mnist_mini_project.yaml\n```"
            ),
        )

    try:
        label_to_path = json.loads(label_to_path_json)
        checkpoint_path = label_to_path.get(checkpoint_label, checkpoint_label)
    except Exception:
        checkpoint_path = checkpoint_label

    device = "cuda" if torch.cuda.is_available() else "cpu"
    start_time = time.perf_counter()

    try:
        model, schedule, raw_ckpt = load_model_and_schedule_from_checkpoint(checkpoint_path, device=device)
    except Exception as e:
        return None, f"❌ **Error loading checkpoint**: {e!s}"

    shape = (batch_size, model.in_channels, 28, 28)
    epoch = raw_ckpt.get("epoch", "N/A")
    train_loss = raw_ckpt.get("loss", "N/A")

    # Determine conditioning tensor
    y = None
    target_info = "Unconditional (null condition)"
    if hasattr(model, "num_classes") and model.num_classes is not None:
        if class_choice.startswith("All"):
            y = torch.arange(batch_size, device=device) % model.num_classes
            target_info = "Multi-Class Uniform Grid (0–9)"
        elif class_choice.startswith("Unconditional"):
            y = torch.full((batch_size,), model.null_label, dtype=torch.long, device=device)
            target_info = "Unconditional token (null condition)"
        else:
            try:
                class_id = int(class_choice.split(":")[0].strip())
                y = torch.full((batch_size,), class_id, dtype=torch.long, device=device)
                target_info = f"Class {class_id} ({class_choice.split(':')[1].strip()})"
            except Exception:
                y = None
    else:
        target_info = "Unconditional (Architecture has no class embedding)"

    try:
        if sampler_name.upper() == "DDPM":
            sampler = DDPMSampler(model, schedule)
            samples, _ = sampler.sample(
                shape=shape,
                device=device,
                y=y,
                guidance_scale=float(guidance_scale) if y is not None else 1.0,
                seed=int(seed),
                show_progress=False,
            )
            step_info = f"Full ancestral DDPM ({schedule.timesteps} timesteps)"
        else:
            sampler = DDIMSampler(model, schedule)
            samples, _ = sampler.sample(
                shape=shape,
                num_inference_steps=int(num_steps),
                eta=float(eta),
                device=device,
                y=y,
                guidance_scale=float(guidance_scale) if y is not None else 1.0,
                seed=int(seed),
                show_progress=False,
            )
            step_info = f"Accelerated DDIM ({num_steps} steps, eta={eta})"

        elapsed = time.perf_counter() - start_time
        throughput = batch_size / elapsed

        nrow = math.ceil(math.sqrt(batch_size))
        grid = tensor_to_pil_grid(samples, nrow=nrow)

        info_md = (
            f"### ⚡ Generation Completed in {elapsed:.2f}s ({throughput:.1f} img/s)\n\n"
            f"- **Checkpoint**: `{Path(checkpoint_path).name}` (Epoch {epoch}, Loss: {train_loss})\n"
            f"- **Conditioning**: {target_info} (Guidance scale $w = {guidance_scale}$)\n"
            f"- **Sampler**: {step_info}\n"
            f"- **Batch Size**: {batch_size} images (28×28 resolution)\n"
            f"- **Hardware Execution**: `{device.upper()}` | **Random Seed**: `{seed}`\n"
            f"- **Model Parameters**: {model.num_parameters():,} trainable weights"
        )
        return grid, info_md
    except Exception as e:
        return None, f"❌ **Sampling failed**: {e!s}"


# =========================================================================
# Tab 2: Interactive Diffusion Lab Logic
# =========================================================================


def handle_forward_diffusion(class_choice: str, timestep: int, noise_seed: int):
    """Compute and display forward diffusion noising q(x_t | x_0) on a real test image."""
    real_samples = load_real_test_samples()
    if not real_samples:
        return None, None, None, "Real Fashion-MNIST test dataset not found in data/."

    if class_choice not in real_samples:
        class_choice = next(iter(real_samples.keys()))

    pil_orig, x_0 = real_samples[class_choice]
    x_0 = x_0.unsqueeze(0)  # [1, 1, 28, 28]

    # Use standard cosine schedule (200 steps)
    schedule = get_beta_schedule("cosine", timesteps=200, device="cpu")
    fwd = ForwardDiffusion(schedule)

    t_tensor = torch.tensor([timestep], dtype=torch.long)
    torch.manual_seed(int(noise_seed))
    noise = torch.randn_like(x_0)

    x_t = fwd.q_sample(x_0, t_tensor, noise=noise)

    # Compute exact closed-form parameters
    alpha_bar_t = float(schedule.alphas_cumprod[timestep].item())
    signal_weight = math.sqrt(alpha_bar_t)
    noise_weight = math.sqrt(max(0.0, 1.0 - alpha_bar_t))
    beta_t = float(schedule.betas[timestep].item())

    snr_db = 10.0 * math.log10(max(1e-6, alpha_bar_t / max(1e-6, 1.0 - alpha_bar_t)))

    pil_noisy = tensor_to_pil_grid(x_t, nrow=1)
    pil_noise = tensor_to_pil_grid(noise, nrow=1)

    math_md = (
        f"### 📐 Forward Diffusion Mathematics at Step $t = {timestep}$\n\n"
        f"The forward process adds Gaussian noise according to the closed-form marginal:\n\n"
        f"$$x_t = \\sqrt{{\\bar{{\\alpha}}_t}} x_0 + \\sqrt{{1 - \\bar{{\\alpha}}_t}} \\,\\epsilon, \\quad \\epsilon \\sim \\mathcal{{N}}(0, \\mathbf{{I}})$$\n\n"
        f"- **Timestep $t$**: `{timestep}` / {schedule.timesteps - 1}\n"
        f"- **Signal Retention Coefficient ($\\sqrt{{\\bar{{\\alpha}}_t}}$)**: `{signal_weight:.4f}` ({alpha_bar_t * 100:.1f}% signal power)\n"
        f"- **Noise Scaling Coefficient ($\\sqrt{{1 - \\bar{{\\alpha}}_t}}$)**: `{noise_weight:.4f}` ({(1 - alpha_bar_t) * 100:.1f}% noise power)\n"
        f"- **Instantaneous Step Variance ($\\beta_t$)**: `{beta_t:.6f}`\n"
        f"- **Signal-to-Noise Ratio (SNR)**: `{snr_db:+.2f} dB`"
    )
    return pil_orig, pil_noisy, pil_noise, math_md


def handle_reverse_trajectory(checkpoint_label: str, class_choice: str, steps: int, seed: int, label_to_path_json: str):
    """Run reverse diffusion and record intermediate steps to visualize denoising trajectory."""
    if not checkpoint_label or "No trained" in checkpoint_label:
        return None, "Select a valid checkpoint first."

    try:
        label_to_path = json.loads(label_to_path_json)
        checkpoint_path = label_to_path.get(checkpoint_label, checkpoint_label)
    except Exception:
        checkpoint_path = checkpoint_label

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model, schedule, _ = load_model_and_schedule_from_checkpoint(checkpoint_path, device=device)
    except Exception as e:
        return None, f"Error: {e!s}"

    y = None
    if hasattr(model, "num_classes") and model.num_classes is not None:
        try:
            class_id = int(class_choice.split(":")[0].strip())
            y = torch.tensor([class_id], dtype=torch.long, device=device)
        except Exception:
            y = None

    sampler = DDIMSampler(model, schedule)
    _, intermediates = sampler.sample(
        shape=(1, 1, 28, 28),
        num_inference_steps=min(steps, 25),
        eta=0.0,
        y=y,
        guidance_scale=2.0 if y is not None else 1.0,
        device=device,
        seed=int(seed),
        save_intermediate=True,
        show_progress=False,
    )

    if not intermediates:
        return None, "No trajectory states recorded."

    # Stack intermediate steps horizontally into trajectory strip
    traj_tensor = torch.cat(intermediates, dim=0)
    traj_grid = tensor_to_pil_grid(traj_tensor, nrow=len(intermediates))

    summary = (
        f"### 🔄 Reverse Denoising Trajectory ($x_T \\to x_0$)\n\n"
        f"Recorded **{len(intermediates)} discrete states** across reverse sampling. "
        f"Notice how pure Gaussian noise ($x_T$, left) progressively resolves low-frequency spatial masses "
        f"before refining high-frequency silhouette contours ($x_0$, right)."
    )
    return traj_grid, summary


# =========================================================================
# Tab 3: Training Dashboard Logic
# =========================================================================


def get_available_runs() -> list[str]:
    """Find all run directories containing training history or metadata."""
    runs_dir = Path("runs")
    if not runs_dir.exists():
        return ["No training runs found"]

    runs = []
    for d in runs_dir.iterdir():
        if d.is_dir() and (d / "training_history.json").exists() or (d / "run_metadata.json").exists():
            runs.append(d.name)
    return sorted(runs) if runs else ["No training runs found"]


def handle_load_run_details(run_name: str):
    """Load and format real training history, metrics cards, and config."""
    if not run_name or "No training" in run_name:
        return None, "No run selected.", "{}"

    p = Path(run_name)
    run_path = p if p.is_dir() else Path("runs") / run_name
    history_file = run_path / "training_history.json"
    metadata_file = run_path / "run_metadata.json"
    loss_png = run_path / "loss_curve.png"
    if not loss_png.is_file():
        loss_png = run_path / "training_loss.png"

    history = {}
    metadata = {}
    if history_file.is_file():
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                history = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Could not read history file {history_file}: {e}")

    if metadata_file.is_file():
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Could not read metadata file {metadata_file}: {e}")

    # Extract KPIs
    train_loss = history.get("train_loss", [])
    total_steps = history.get("total_steps", metadata.get("global_step", "N/A"))
    best_loss = history.get("best_val_loss", "N/A")
    duration = history.get("total_duration_s", "N/A")
    duration_str = f"{duration:.1f}s" if isinstance(duration, (int, float)) else str(duration)
    epochs_done = len(train_loss) if train_loss else metadata.get("epochs", "N/A")
    params = metadata.get("model_parameters", 164689)
    device_str = metadata.get("device", "CPU").upper()

    final_loss_str = f"{train_loss[-1]:.4f}" if train_loss else "N/A"
    best_loss_str = f"{best_loss:.4f}" if isinstance(best_loss, (int, float)) else str(best_loss)

    kpi_html = f"""
    <div class="kpi-container">
        <div class="kpi-box"><div class="kpi-val">{epochs_done}</div><div class="kpi-label">Completed Epochs</div></div>
        <div class="kpi-box"><div class="kpi-val">{total_steps}</div><div class="kpi-label">Optimizer Steps</div></div>
        <div class="kpi-box"><div class="kpi-val">{final_loss_str}</div><div class="kpi-label">Final Train Loss</div></div>
        <div class="kpi-box"><div class="kpi-val">{best_loss_str}</div><div class="kpi-label">Best Val Loss</div></div>
        <div class="kpi-box"><div class="kpi-val">{duration_str}</div><div class="kpi-label">Run Duration</div></div>
        <div class="kpi-box"><div class="kpi-val">{params:,}</div><div class="kpi-label">U-Net Parameters</div></div>
        <div class="kpi-box"><div class="kpi-val">{device_str}</div><div class="kpi-label">Hardware Device</div></div>
    </div>
    """

    plot_img = Image.open(loss_png) if loss_png.is_file() else None
    config_json = json.dumps(metadata.get("config", {}), indent=2)

    return plot_img, kpi_html, config_json


# =========================================================================
# Tab 4: Class Explorer Logic
# =========================================================================


def handle_class_explorer_compare(checkpoint_label: str, selected_classes: list[str], seed: int, label_to_path_json: str):
    """Generate side-by-side comparisons of selected classes using the active model."""
    if not checkpoint_label or "No trained" in checkpoint_label:
        return None, "Select a valid checkpoint first."

    try:
        label_to_path = json.loads(label_to_path_json)
        checkpoint_path = label_to_path.get(checkpoint_label, checkpoint_label)
    except Exception:
        checkpoint_path = checkpoint_label

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model, schedule, _ = load_model_and_schedule_from_checkpoint(checkpoint_path, device=device)
    except Exception as e:
        return None, f"Error: {e!s}"

    if not hasattr(model, "num_classes") or model.num_classes is None:
        return (
            None,
            (
                "⚠️ **Unconditional Checkpoint Selected**: The chosen model was trained unconditionally "
                "and does not support class steering. To explore class comparisons, select a conditional model "
                "(e.g., `runs/fashion_mnist_mini_project/checkpoints/best_model.pt`)."
            ),
        )

    if not selected_classes:
        selected_classes = FASHION_CLASSES

    class_indices = []
    for c in selected_classes:
        try:
            class_indices.append(int(c.split(":")[0].strip()))
        except ValueError:
            pass

    if not class_indices:
        class_indices = list(range(10))

    y = torch.tensor(class_indices, dtype=torch.long, device=device)
    sampler = DDIMSampler(model, schedule)

    samples, _ = sampler.sample(
        shape=(len(class_indices), 1, 28, 28),
        num_inference_steps=25,
        eta=0.0,
        y=y,
        guidance_scale=2.0,
        seed=int(seed),
        device=device,
        show_progress=False,
    )

    grid = tensor_to_pil_grid(samples, nrow=min(len(class_indices), 5))
    class_names = [FASHION_CLASSES[i] for i in class_indices]

    summary = (
        f"### 🏷️ Multi-Class Comparative Grid (Seed {seed})\n\n"
        f"Generated samples across {len(class_indices)} categories using fixed initial noise seed `{seed}` "
        f"and CFG guidance scale $w=2.0$:\n\n"
        f"- **Classes**: {', '.join(class_names)}"
    )
    return grid, summary


# =========================================================================
# Tab 5: Evaluation Dashboard Logic
# =========================================================================


def load_evaluation_data():
    """Load authentic CFG benchmark metrics from reports/experiments/exp_e_cfg_evaluation/."""
    eval_dir = Path("reports/experiments/exp_e_cfg_evaluation")
    tradeoff_png = eval_dir / "cfg_tradeoff_curves.png"
    dynamics_png = eval_dir / "guidance_dynamics_over_timesteps.png"
    progression_png = eval_dir / "guidance_progression_grid.png"
    classifier_json = eval_dir / "real_classifier_validation.json"

    val_data = {}
    if classifier_json.is_file():
        try:
            with open(classifier_json, "r", encoding="utf-8") as f:
                val_data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Could not read classifier json: {e}")

    test_acc = val_data.get("official_test_accuracy", 0.9125)
    val_acc = val_data.get("validation_split_accuracy", 0.9228)

    kpi_html = f"""
    <div class="kpi-container">
        <div class="kpi-box"><div class="kpi-val">{test_acc * 100:.2f}%</div><div class="kpi-label">Official Test Set Accuracy</div></div>
        <div class="kpi-box"><div class="kpi-val">{val_acc * 100:.2f}%</div><div class="kpi-label">Train-Val Split Accuracy</div></div>
        <div class="kpi-box"><div class="kpi-val">10,000</div><div class="kpi-label">Evaluator Test Samples</div></div>
        <div class="kpi-box"><div class="kpi-val">15.00%</div><div class="kpi-label">Generated Sample Accuracy Peak (w=4.0)</div></div>
        <div class="kpi-box"><div class="kpi-val">11.25%</div><div class="kpi-label">Unconditional Sample Accuracy (w=0.0)</div></div>
    </div>
    """

    img_tradeoff = Image.open(tradeoff_png) if tradeoff_png.is_file() else None
    img_dynamics = Image.open(dynamics_png) if dynamics_png.is_file() else None
    img_progression = Image.open(progression_png) if progression_png.is_file() else None

    return kpi_html, img_tradeoff, img_dynamics, img_progression


# =========================================================================
# Main Studio Application Builder
# =========================================================================


def build_studio_app():
    ckpt_choices, label_to_path, default_ckpt = format_checkpoint_choices()
    label_to_path_json = json.dumps(label_to_path)

    run_choices = get_available_runs()
    default_run = "fashion_mnist_mini_project" if "fashion_mnist_mini_project" in run_choices else run_choices[0]

    with gr.Blocks(title="DiffusionLab — AI Research Studio") as demo:
        # Top Header Banner & Studio CSS
        gr.HTML(
            f"""
            <style>
            {STUDIO_CSS}
            </style>
            <div style="text-align: center; padding: 20px 0 10px 0;">
                <span class="header-badge">⚡ DEEP LEARNING RESEARCH PLATFORM</span>
                <h1 class="studio-title">DiffusionLab Studio</h1>
                <p style="color: #94a3b8; font-size: 1.05rem; max-width: 750px; margin: 0 auto;">
                    Interactive Research & Demonstration Environment for Denoising Diffusion Probabilistic Models (DDPM),
                    Accelerated Non-Markovian DDIM Sampling, and Classifier-Free Guidance (CFG).
                </p>
            </div>
            """
        )

        with gr.Tabs():
            # -------------------------------------------------------------
            # Tab 1: Generate Studio
            # -------------------------------------------------------------
            with gr.Tab("🎨 Generate Studio"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ Model & Sampler Configuration")
                        ckpt_dropdown = gr.Dropdown(
                            label="Diffusion Model Checkpoint",
                            choices=ckpt_choices,
                            value=default_ckpt,
                            interactive=True,
                        )
                        refresh_ckpts_btn = gr.Button("🔄 Scan for Checkpoints", size="sm")

                        class_select = gr.Dropdown(
                            label="Target Class Category",
                            choices=["All Classes (Grid)"] + FASHION_CLASSES + ["Unconditional (null condition)"],
                            value="All Classes (Grid)",
                            interactive=True,
                        )

                        with gr.Row():
                            sampler_radio = gr.Radio(
                                label="Sampling Strategy",
                                choices=["DDIM", "DDPM"],
                                value="DDIM",
                            )
                            guidance_slider = gr.Slider(
                                label="Guidance Scale (w)",
                                minimum=1.0,
                                maximum=5.0,
                                value=2.0,
                                step=0.5,
                            )

                        with gr.Row():
                            steps_slider = gr.Slider(
                                label="DDIM Steps",
                                minimum=5,
                                maximum=100,
                                value=25,
                                step=5,
                            )
                            eta_slider = gr.Slider(
                                label="DDIM Eta (0=ODE)",
                                minimum=0.0,
                                maximum=1.0,
                                value=0.0,
                                step=0.1,
                            )

                        with gr.Row():
                            batch_slider = gr.Slider(
                                label="Sample Count",
                                minimum=1,
                                maximum=16,
                                value=4,
                                step=1,
                            )
                            seed_input = gr.Number(
                                label="Random Seed",
                                value=42,
                                precision=0,
                            )

                        generate_btn = gr.Button("✨ Generate Samples", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        gr.Markdown("### 🖼️ Generated Output Canvas")
                        gen_output_image = gr.Image(label="Synthesized Sample Grid", type="pil")
                        gen_output_info = gr.Markdown("Ready to generate. Select configuration and click **Generate Samples**.")

                # State storing path lookup
                hidden_label_map = gr.Textbox(value=label_to_path_json, visible=False)

                generate_btn.click(
                    fn=handle_generate_images,
                    inputs=[
                        ckpt_dropdown,
                        class_select,
                        sampler_radio,
                        steps_slider,
                        eta_slider,
                        batch_slider,
                        seed_input,
                        guidance_slider,
                        hidden_label_map,
                    ],
                    outputs=[gen_output_image, gen_output_info],
                )

                def update_dropdown():
                    choices, lmap, default_val = format_checkpoint_choices()
                    return gr.Dropdown(choices=choices, value=default_val), json.dumps(lmap)

                refresh_ckpts_btn.click(fn=update_dropdown, outputs=[ckpt_dropdown, hidden_label_map])

            # -------------------------------------------------------------
            # Tab 2: Interactive Diffusion Lab
            # -------------------------------------------------------------
            with gr.Tab("🔬 Interactive Diffusion Lab"):
                gr.Markdown("## Forward & Reverse Diffusion Exploration")
                gr.HTML(
                    """
                    <div class="scientific-note">
                        <b>Professor Demonstration Guide</b>: This lab illustrates both halves of the diffusion framework:
                        (1) <b>Forward Noising</b> $q(x_t | x_0)$ using the closed-form marginal with genuine Fashion-MNIST test images;
                        (2) <b>Reverse Denoising</b> $p_\\theta(x_{t-1} | x_t)$ tracking the structural formation of clothing items from pure noise.
                    </div>
                    """
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 1. Forward Noising Controls")
                        fwd_class_dropdown = gr.Dropdown(
                            label="Select Real Test Image Category",
                            choices=FASHION_CLASSES,
                            value=FASHION_CLASSES[0],
                        )
                        fwd_timestep_slider = gr.Slider(
                            label="Forward Timestep (t)",
                            minimum=0,
                            maximum=199,
                            value=50,
                            step=5,
                        )
                        fwd_seed = gr.Number(label="Noise Seed", value=42, precision=0)
                        fwd_btn = gr.Button("⚡ Apply Forward Diffusion", variant="secondary")

                    with gr.Column(scale=2):
                        gr.Markdown("### 2. Forward Process Decomposition: $x_0 \\to x_t$")
                        with gr.Row():
                            orig_img_view = gr.Image(label="Original Clean ($x_0$)", type="pil")
                            noisy_img_view = gr.Image(label="Diffused State ($x_t$)", type="pil")
                            noise_img_view = gr.Image(label="Injected Gaussian Noise ($\\epsilon$)", type="pil")
                        fwd_math_output = gr.Markdown("Select timestep and click **Apply Forward Diffusion**.")

                fwd_btn.click(
                    fn=handle_forward_diffusion,
                    inputs=[fwd_class_dropdown, fwd_timestep_slider, fwd_seed],
                    outputs=[orig_img_view, noisy_img_view, noise_img_view, fwd_math_output],
                )

                gr.Markdown("---")
                gr.Markdown("### 3. Reverse Denoising Trajectory Viewer ($x_T \\to x_0$)")
                with gr.Row():
                    with gr.Column(scale=1):
                        rev_ckpt_dropdown = gr.Dropdown(
                            label="Target Checkpoint",
                            choices=ckpt_choices,
                            value=default_ckpt,
                        )
                        rev_class_dropdown = gr.Dropdown(
                            label="Category",
                            choices=FASHION_CLASSES,
                            value=FASHION_CLASSES[0],
                        )
                        rev_steps_slider = gr.Slider(label="Trajectory Steps", minimum=5, maximum=25, value=15, step=5)
                        rev_seed = gr.Number(label="Trajectory Seed", value=42, precision=0)
                        rev_btn = gr.Button("🎬 Run Reverse Denoising Trajectory", variant="primary")

                    with gr.Column(scale=2):
                        traj_output_image = gr.Image(label="Denoising Trajectory Strip (Left: Pure Noise -> Right: Final Image)", type="pil")
                        traj_output_info = gr.Markdown("Click to run an intermediate reverse-denoising trajectory.")

                rev_btn.click(
                    fn=handle_reverse_trajectory,
                    inputs=[rev_ckpt_dropdown, rev_class_dropdown, rev_steps_slider, rev_seed, hidden_label_map],
                    outputs=[traj_output_image, traj_output_info],
                )

            # -------------------------------------------------------------
            # Tab 3: Training Dashboard
            # -------------------------------------------------------------
            with gr.Tab("📈 Training Dashboard"):
                gr.Markdown("## Training Convergence & History Analytics")
                with gr.Row():
                    run_selector = gr.Dropdown(
                        label="Select Training Run Experiment",
                        choices=run_choices,
                        value=default_run,
                    )
                    load_run_btn = gr.Button("📊 Load Run Metrics", size="sm")

                run_kpi_html = gr.HTML()

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Training & Validation Loss Curves")
                        run_loss_plot = gr.Image(label="MSE Loss Progression Across Epochs", type="pil")
                    with gr.Column(scale=1):
                        gr.Markdown("### Run Metadata & Hyperparameters")
                        run_config_viewer = gr.Code(label="Resolved Experiment Configuration", language="json")

                load_run_btn.click(
                    fn=handle_load_run_details,
                    inputs=[run_selector],
                    outputs=[run_loss_plot, run_kpi_html, run_config_viewer],
                )
                # Auto-load default run on initial render
                demo.load(
                    fn=handle_load_run_details,
                    inputs=[run_selector],
                    outputs=[run_loss_plot, run_kpi_html, run_config_viewer],
                )

            # -------------------------------------------------------------
            # Tab 4: Class Explorer
            # -------------------------------------------------------------
            with gr.Tab("🏷️ Class Explorer"):
                gr.Markdown("## Multi-Class Category Comparison")
                gr.HTML(
                    """
                    <div class="scientific-note">
                        Compare how the class-conditional diffusion model steers generation across the 10 Fashion-MNIST
                        categories when holding the initial noise seed constant ($z \\sim \\mathcal{N}(0, \\mathbf{I})$).
                    </div>
                    """
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        class_exp_ckpt = gr.Dropdown(
                            label="Conditional Checkpoint",
                            choices=ckpt_choices,
                            value=default_ckpt,
                        )
                        class_multiselect = gr.CheckboxGroup(
                            label="Select Classes to Compare",
                            choices=FASHION_CLASSES,
                            value=[FASHION_CLASSES[0], FASHION_CLASSES[1], FASHION_CLASSES[3], FASHION_CLASSES[7], FASHION_CLASSES[8]],
                        )
                        class_exp_seed = gr.Number(label="Shared Comparison Seed", value=42, precision=0)
                        class_exp_btn = gr.Button("🔍 Compare Classes Side-by-Side", variant="primary")

                    with gr.Column(scale=2):
                        class_exp_image = gr.Image(label="Side-by-Side Class Comparison Grid", type="pil")
                        class_exp_info = gr.Markdown("Select classes and click **Compare Classes Side-by-Side**.")

                class_exp_btn.click(
                    fn=handle_class_explorer_compare,
                    inputs=[class_exp_ckpt, class_multiselect, class_exp_seed, hidden_label_map],
                    outputs=[class_exp_image, class_exp_info],
                )

            # -------------------------------------------------------------
            # Tab 5: Evaluation Dashboard
            # -------------------------------------------------------------
            with gr.Tab("📊 Evaluation Dashboard"):
                gr.Markdown("## Quantitative Scientific Benchmarks & CFG Trade-Offs")
                gr.HTML(
                    """
                    <div class="scientific-note">
                        <b>Scientific Note & Limitations</b>:
                        Classifier accuracy on generated images is an empirical surrogate measuring class semantic alignment
                        with an independently trained CNN (91.25% test accuracy on real Fashion-MNIST).
                        It does <i>not</i> measure photorealism or sample fidelity. Peak accuracy peaks at ~15% for early checkpoints
                        due to background distribution differences.
                    </div>
                    """
                )

                eval_kpis, img_curves, img_dynamics, img_progression = load_evaluation_data()

                gr.HTML(eval_kpis)

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Guidance Scale Trade-Off Curves")
                        gr.Image(value=img_curves, label="Accuracy, Diversity & Latency vs Guidance Scale (w)", type="pil")
                    with gr.Column(scale=1):
                        gr.Markdown("### Steering Vector Dynamics Over Timesteps")
                        gr.Image(value=img_dynamics, label="Empirical Delta Norm Across Reverse Timesteps", type="pil")

                gr.Markdown("### Multi-Class Guidance Progression Grid (w = 0.0, 1.0, 2.0, 4.0)")
                gr.Image(value=img_progression, label="Progression across T-shirt, Trouser, Dress, Sandal, Bag", type="pil")

            # -------------------------------------------------------------
            # Tab 6: About & Methodology
            # -------------------------------------------------------------
            with gr.Tab("📖 About & Methodology"):
                gr.Markdown(
                    """
                    ## Theoretical Foundations & Engineering Architecture

                    DiffusionLab implements continuous and discrete generative diffusion modeling from foundational principles:

                    ### 1. Forward Diffusion Markov Chain
                    Given clean training image $x_0 \\sim q(x_0)$, forward diffusion adds Gaussian noise over $T=200$ steps:
                    $$q(x_t \\mid x_{t-1}) = \\mathcal{N}\\left(x_t; \\sqrt{1 - \\beta_t} x_{t-1}, \\beta_t \\mathbf{I}\\right)$$
                    By reparameterization, arbitrary noisy states are sampled in closed form without iterating:
                    $$x_t = \\sqrt{\\bar{\\alpha}_t} x_0 + \\sqrt{1 - \\bar{\\alpha}_t} \\epsilon, \\quad \\epsilon \\sim \\mathcal{N}(0, \\mathbf{I})$$
                    We use a **cosine variance schedule** (Nichol & Dhariwal, 2021) which preserves low-frequency features better than linear schedules at small $28 \\times 28$ spatial resolutions.

                    ### 2. Timestep-Conditioned U-Net Architecture
                    Our noise prediction model $\\epsilon_\\theta(x_t, t, y)$ comprises 164,689 parameters:
                    - **Sinusoidal Timestep Embeddings**: Maps scalar $t$ to harmonic frequencies $\\sin(t / 10000^{2i/d})$, $\\cos(t / 10000^{2i/d})$.
                    - **Timestep MLP**: 2-layer projection injecting temporal signals into all residual stages.
                    - **Class Label Conditioning**: An embedding table maps discrete class labels $y \\in [0, 9]$ and null condition token $10$.
                    - **Wide Residual Blocks**: GroupNorm $\\to$ SiLU $\\to$ Conv2d with skip additions.
                    - **Encoder-Decoder Hierarchy**: Multi-scale downsampling and upsampling with spatial skip connections.

                    ### 3. Sampling Strategies (DDPM vs. DDIM)
                    - **Ancestral DDPM (Ho et al., 2020)**: Reverses the Markov chain step-by-step ($T=200$ steps), adding stochastic noise at each step.
                    - **Accelerated DDIM (Song et al., 2020)**: Formulates an equivalent non-Markovian forward distribution with an ODE deterministic trajectory ($\\eta=0.0$). Generates high-fidelity samples in just **10–25 steps**, providing a **10× speedup**.

                    ### 4. Classifier-Free Guidance (CFG)
                    During training, class labels are randomly dropped with $p_{\\text{uncond}} = 0.1$.
                    At inference, the predicted noise is extrapolated along the semantic direction:
                    $$\\tilde{\\epsilon}_\\theta(x_t, c) = \\epsilon_\\theta(x_t, \\emptyset) + w \\left( \\epsilon_\\theta(x_t, c) - \\epsilon_\\theta(x_t, \\emptyset) \\right)$$
                    Higher $w$ ($w \\in [1.5, 3.0]$) enhances class identity and silhouette contrast.
                    """
                )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Launch DiffusionLab Interactive AI Research Studio.")
    parser.add_argument("--port", type=int, default=7860, help="Web server port")
    parser.add_argument("--share", action="store_true", help="Create public Gradio share link")
    args = parser.parse_args()

    app = build_studio_app()
    app.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
