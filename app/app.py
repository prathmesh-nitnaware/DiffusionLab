"""Gradio web demonstration interface for DiffusionLab."""

import argparse
import sys
from pathlib import Path

import gradio as gr
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffusionlab.data.transforms import tensor_to_pil_grid
from diffusionlab.diffusion.samplers import DDIMSampler, DDPMSampler


def find_available_checkpoints() -> list[str]:
    """Search for checkpoint files in runs/ and checkpoints/."""
    ckpts = []
    for search_dir in ["runs", "checkpoints"]:
        p = Path(search_dir)
        if p.exists():
            for pt in p.rglob("*.pt"):
                ckpts.append(str(pt))
    return sorted(ckpts)


def load_model_from_checkpoint(checkpoint_path: str, device: str = "cpu"):
    """Instantiate and load model from checkpoint."""
    from diffusionlab.inference.loader import load_model_and_schedule_from_checkpoint

    return load_model_and_schedule_from_checkpoint(checkpoint_path, device=device)


def generate_images(
    checkpoint_choice: str,
    sampler_name: str,
    num_steps: int,
    eta: float,
    batch_size: int,
    seed: int,
    class_choice: str = "All Classes (Grid)",
    guidance_scale: float = 2.0,
):
    """Generate image grid from selected checkpoint and options."""
    if not checkpoint_choice or checkpoint_choice == "None available":
        return None, "Error: Please select or train a valid checkpoint first."

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model, schedule, raw_ckpt = load_model_from_checkpoint(checkpoint_choice, device=device)
    except Exception as e:
        return None, f"Error loading checkpoint: {e!s}"

    shape = (batch_size, model.in_channels, 28, 28)
    epoch = raw_ckpt.get("epoch", "N/A")
    train_loss = raw_ckpt.get("loss", "N/A")

    # Handle class conditioning
    y = None
    target_class_info = "Unconditional"
    if hasattr(model, "num_classes") and model.num_classes is not None:
        if class_choice.startswith("All"):
            y = torch.arange(batch_size, device=device) % model.num_classes
            target_class_info = "Multi-Class Grid (0-9)"
        elif class_choice.startswith("Unconditional"):
            y = torch.full((batch_size,), model.null_label, dtype=torch.long, device=device)
            target_class_info = "Unconditional (null condition)"
        else:
            try:
                class_id = int(class_choice.split(":")[0].strip())
                y = torch.full((batch_size,), class_id, dtype=torch.long, device=device)
                target_class_info = f"Class {class_id} ({class_choice.split(':')[1].strip()})"
            except Exception:
                y = None

    try:
        if sampler_name.upper() == "DDPM":
            sampler = DDPMSampler(model, schedule)
            samples, _ = sampler.sample(
                shape=shape,
                device=device,
                y=y,
                guidance_scale=float(guidance_scale),
                seed=seed,
                show_progress=False,
            )
            step_info = f"Ancestral DDPM ({schedule.timesteps} steps)"
        else:
            sampler = DDIMSampler(model, schedule)
            samples, _ = sampler.sample(
                shape=shape,
                num_inference_steps=int(num_steps),
                eta=float(eta),
                device=device,
                y=y,
                guidance_scale=float(guidance_scale),
                seed=seed,
                show_progress=False,
            )
            step_info = f"Accelerated DDIM ({num_steps} steps, eta={eta})"

        grid = tensor_to_pil_grid(samples, nrow=int(batch_size**0.5) if batch_size > 1 else 1)

        info = (
            f"### Generation Summary\n"
            f"- **Project**: DiffusionLab — Fashion Image Generation Using DDPM\n"
            f"- **Checkpoint**: `{Path(checkpoint_choice).name}`\n"
            f"- **Checkpoint Epoch**: {epoch} | **Loss**: {train_loss}\n"
            f"- **Target Conditioning**: {target_class_info} (Guidance scale: {guidance_scale})\n"
            f"- **Sampling Strategy**: {step_info}\n"
            f"- **Random Seed**: {seed}\n"
            f"- **Execution Device**: `{device.upper()}`\n"
            f"- **Model Parameters**: {model.num_parameters():,}"
        )
        return grid, info
    except Exception as e:
        return None, f"Generation failed: {e!s}"


def build_app():
    available_ckpts = find_available_checkpoints()
    default_ckpt = available_ckpts[0] if available_ckpts else "None available"

    fashion_classes = [
        "All Classes (Grid)",
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
        "Unconditional (null condition)",
    ]

    with gr.Blocks(title="DiffusionLab — Fashion Image Generation Using DDPM") as demo:
        gr.Markdown(
            """
            # 👗 DiffusionLab — Fashion Image Generation Using DDPM
            ### College Mini-Project Demonstration
            Generate Fashion-MNIST clothing images at 28×28 resolution using custom Denoising Diffusion Probabilistic Models (DDPM) and accelerated DDIM sampling.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Model & Sampler Configuration")

                ckpt_dropdown = gr.Dropdown(
                    label="Model Checkpoint",
                    choices=available_ckpts if available_ckpts else ["None available"],
                    value=default_ckpt,
                    interactive=True,
                )

                refresh_btn = gr.Button("🔄 Scan for Checkpoints", size="sm")

                class_dropdown = gr.Dropdown(
                    label="Fashion-MNIST Class",
                    choices=fashion_classes,
                    value="All Classes (Grid)",
                    interactive=True,
                )

                guidance_slider = gr.Slider(
                    label="CFG Guidance Scale (1.0 = Standard, >1.0 = Guided)",
                    minimum=1.0,
                    maximum=5.0,
                    value=2.0,
                    step=0.5,
                )

                sampler_radio = gr.Radio(
                    label="Sampling Strategy",
                    choices=["DDIM", "DDPM"],
                    value="DDIM",
                )

                steps_slider = gr.Slider(
                    label="DDIM Inference Steps",
                    minimum=5,
                    maximum=200,
                    value=25,
                    step=5,
                )

                eta_slider = gr.Slider(
                    label="DDIM Eta (0.0 = Deterministic ODE, 1.0 = Stochastic)",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0,
                    step=0.1,
                )

                batch_slider = gr.Slider(
                    label="Batch Size (Number of Images)",
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

                generate_btn = gr.Button("✨ Generate Samples", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("### 🖼️ Generated Output")
                output_image = gr.Image(label="Generated Sample Grid", type="pil")
                output_info = gr.Markdown("Ready to generate. Select options and click 'Generate Samples'.")

        def update_ckpts():
            ckpts = find_available_checkpoints()
            return gr.Dropdown(choices=ckpts if ckpts else ["None available"], value=ckpts[0] if ckpts else "None available")

        refresh_btn.click(fn=update_ckpts, outputs=ckpt_dropdown)

        generate_btn.click(
            fn=generate_images,
            inputs=[
                ckpt_dropdown,
                sampler_radio,
                steps_slider,
                eta_slider,
                batch_slider,
                seed_input,
                class_dropdown,
                guidance_slider,
            ],
            outputs=[output_image, output_info],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Launch DiffusionLab Gradio Web Interface.")
    parser.add_argument("--port", type=int, default=7860, help="Web server port")
    parser.add_argument("--share", action="store_true", help="Create Gradio share link")
    args = parser.parse_args()

    app = build_app()
    app.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
