"""
Gradio demo for the NIH CXR-14 classifier.
Runs locally or as a Hugging Face Space.

Upload a chest X-ray → get 14 disease probabilities, a Grad-CAM overlay,
and a plain-text summary of the top predictions.
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless backend — no display needed in Spaces
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# ── path setup ────────────────────────────────────────────────────────────────
# This file lives at the project root; src/ modules are one level down.
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset import DISEASE_LABELS, get_val_transforms
from model   import build_model
from evaluate import GradCAM, _denormalize

# ── checkpoint loading ────────────────────────────────────────────────────────

# Where to look for the checkpoint locally (works when running from project root)
LOCAL_CKPT = PROJECT_ROOT / "outputs" / "checkpoints" / "best_model.pth"

# If the checkpoint is not local, attempt to download from Hugging Face Hub.
# Set this to your own model repo, e.g. "your-username/chest-xray-densenet121"
# Leave as None to skip Hub download and rely on the local file.
HF_MODEL_REPO     = os.environ.get("HF_MODEL_REPO", None)
HF_MODEL_FILENAME = "best_model.pth"


def _get_device() -> torch.device:
    """Return CUDA → MPS → CPU, whichever is available."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_checkpoint() -> Path:
    """Return path to checkpoint, downloading from HF Hub if not found locally."""
    if LOCAL_CKPT.exists():
        return LOCAL_CKPT

    if HF_MODEL_REPO:
        try:
            from huggingface_hub import hf_hub_download
            print(f"Downloading checkpoint from HF Hub: {HF_MODEL_REPO} …")
            downloaded = hf_hub_download(
                repo_id=HF_MODEL_REPO,
                filename=HF_MODEL_FILENAME,
                # cache_dir puts it in ~/.cache/huggingface on Spaces
            )
            return Path(downloaded)
        except Exception as e:
            raise FileNotFoundError(
                f"Could not download checkpoint from HF Hub ({HF_MODEL_REPO}): {e}"
            ) from e

    raise FileNotFoundError(
        f"No checkpoint found at {LOCAL_CKPT}.\n"
        "Train the model first (`python src/train.py`) or set HF_MODEL_REPO "
        "to download from Hugging Face Hub."
    )


# ── load model once at startup ────────────────────────────────────────────────
# Models are loaded once when the Space starts, not on every request.
# This is much faster than reloading weights for each user.

DEVICE = _get_device()
print(f"Demo device: {DEVICE}")

try:
    ckpt_path = _load_checkpoint()
    _model = build_model()
    _ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    _model.load_state_dict(_ckpt["model_state_dict"])
    _model.to(DEVICE)
    _model.eval()
    print(f"Model loaded from {ckpt_path.name}  "
          f"(epoch {_ckpt.get('epoch','?')}, "
          f"val_loss {_ckpt.get('val_loss', float('nan')):.4f})")
    MODEL_READY = True
except FileNotFoundError as e:
    print(f"WARNING: {e}")
    _model     = None
    MODEL_READY = False

TRANSFORM = get_val_transforms()


# ── inference helpers ─────────────────────────────────────────────────────────

def _build_prob_chart(probs: np.ndarray) -> Image.Image:
    """Render a horizontal bar chart of all 14 probabilities as a PIL Image."""
    import io

    sorted_idx    = np.argsort(probs)          # ascending
    labels_sorted = [DISEASE_LABELS[i] for i in sorted_idx]
    probs_sorted  = probs[sorted_idx]

    colours = ["#e05c5c" if p > 0.5 else "steelblue" for p in probs_sorted]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh(labels_sorted, probs_sorted, color=colours, edgecolor="white")
    ax.set_xlim(0, 1)
    ax.axvline(0.5, color="red", linestyle="--", linewidth=0.8, alpha=0.6,
               label="0.5 threshold")
    ax.set_xlabel("Predicted probability")
    ax.set_title("Model predictions (all 14 conditions)")
    ax.legend(fontsize=8)

    for bar, p in zip(bars, probs_sorted):
        ax.text(min(p + 0.02, 0.97), bar.get_y() + bar.get_height() / 2,
                f"{p:.2f}", va="center", fontsize=7)

    plt.tight_layout()

    # Render to a PIL Image via an in-memory PNG buffer
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()   # .copy() detaches from the BytesIO buffer


def _build_gradcam_overlay(
    tensor:       torch.Tensor,   # (1, 3, 224, 224) on DEVICE
    target_class: int,
) -> np.ndarray:
    """Run Grad-CAM and return an RGB uint8 blend (224, 224, 3)."""
    grad_cam = GradCAM(_model, target_layer_name="features.denseblock4")
    cam, _, _ = grad_cam.generate(tensor, target_class=target_class)
    grad_cam.remove_hooks()

    # Denormalise original image
    original = _denormalize(tensor.squeeze(0))   # (224, 224, 3) uint8

    # Colour the CAM and blend
    colormap = cm.get_cmap("jet")
    cam_rgb  = (colormap(cam)[:, :, :3] * 255).astype(np.uint8)
    blend    = (0.55 * original + 0.45 * cam_rgb).astype(np.uint8)
    return blend


# ── main prediction function ──────────────────────────────────────────────────

def predict(uploaded_image: Image.Image):
    """Run inference and return (gradcam_overlay, prob_chart, summary_text)."""
    if not MODEL_READY:
        error_msg = (
            "Model checkpoint not found. "
            "Please train the model first or configure HF_MODEL_REPO."
        )
        # Return empty outputs with error message
        return None, None, error_msg

    image_rgb = uploaded_image.convert("RGB")
    tensor    = TRANSFORM(image_rgb).unsqueeze(0).to(DEVICE)   # (1,3,224,224)

    with torch.no_grad():
        logits = _model(tensor)              # (1, 14)
        probs  = torch.sigmoid(logits)       # (1, 14) in [0,1]

    probs_np = probs.cpu().numpy()[0]        # (14,)

    # Top predicted class drives the Grad-CAM explanation
    top_class = int(np.argmax(probs_np))

    blend_np     = _build_gradcam_overlay(tensor, top_class)
    gradcam_pil  = Image.fromarray(blend_np)

    prob_fig = _build_prob_chart(probs_np)

    top3_idx = np.argsort(probs_np)[::-1][:3]
    lines    = ["Top predictions:"]
    for i, idx in enumerate(top3_idx, 1):
        lines.append(f"  {i}. {DISEASE_LABELS[idx]}: {probs_np[idx]:.1%}")
    lines.append("")
    lines.append(
        "DISCLAIMER: Research tool only. Not for clinical use."
    )
    summary = "\n".join(lines)

    return gradcam_pil, prob_fig, summary


# ── Gradio interface ──────────────────────────────────────────────────────────

import gradio as gr

EXAMPLES_DIR = PROJECT_ROOT / "outputs" / "figures"
example_files = sorted(EXAMPLES_DIR.glob("gradcam_*.png"))[:2]
examples = [[str(p)] for p in example_files] if example_files else None

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(
        type="pil",
        label="Upload a chest X-ray (PA or AP view)",
    ),
    outputs=[
        gr.Image(
            type="pil",
            label="Grad-CAM — regions driving the top prediction",
        ),
        gr.Image(
            type="pil",
            label="Predicted probabilities for all 14 conditions",
        ),
        gr.Textbox(
            label="Summary",
            lines=6,
        ),
    ],
    title="Chest X-Ray Disease Classifier",
    description=(
        "Upload a frontal chest X-ray to get predictions for 14 thoracic conditions "
        "using a fine-tuned DenseNet-121 trained on the NIH Chest X-Ray 14 dataset.\n\n"
        "The Grad-CAM overlay highlights which regions of the X-ray the model focused on "
        "when making its top prediction (red = most influential).\n\n"
        "**This is a research tool. Do not use for medical diagnosis.**"
    ),
    examples=examples,
    allow_flagging="never",
    # gr.themes was added in Gradio 4 — omit for 3.x compatibility
)

if __name__ == "__main__":
    # share=False works fine in gradio 4.x on Mac (the localhost-inaccessible
    # error only affected certain 3.x builds).  Set share=True if you want a
    # temporary public URL via Gradio's tunnel service.
    demo.launch(share=False, server_name="127.0.0.1", show_error=True)
