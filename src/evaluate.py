"""
Evaluate the trained model on the NIH CXR-14 test set.

Outputs: per-class AUC-ROC table, ROC curve figure, and Grad-CAM overlays.

Usage:
    python src/evaluate.py
    python src/evaluate.py --ckpt outputs/checkpoints/last_model.pth
    python src/evaluate.py --n-gradcam 10
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import (
    DISEASE_LABELS,
    NUM_CLASSES,
    build_dataloaders,
    build_image_index,
    get_val_transforms,
    RAW_DIR,
)
from model import build_model

# ── paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR     = PROJECT_ROOT / "outputs" / "checkpoints"
FIGURES_DIR  = PROJECT_ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CKPT = CKPT_DIR / "best_model.pth"


# ── device ────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD CHECKPOINT
# ══════════════════════════════════════════════════════════════════════════════

def load_model(ckpt_path: Path, device: torch.device) -> nn.Module:
    """Load DenseNet-121 weights from a checkpoint."""
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Run src/train.py first to generate a checkpoint."
        )

    model = build_model()
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()   # disables Dropout for deterministic inference

    epoch    = ckpt.get("epoch", "?")
    val_loss = ckpt.get("val_loss", float("nan"))
    print(f"Loaded checkpoint: {ckpt_path.name}  (epoch {epoch}, val_loss {val_loss:.4f})")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 2. COLLECT PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (probs, labels) arrays over the full DataLoader."""
    all_probs  = []
    all_labels = []

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)

        logits = model(images)               # (B, 14) — raw logits
        probs  = torch.sigmoid(logits)       # (B, 14) — probabilities in [0,1]

        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())    # labels were already on CPU

        if (batch_idx + 1) % 50 == 0:
            print(f"  Inference: {batch_idx+1}/{len(loader)} batches", end="\r")

    print()
    probs_arr  = np.concatenate(all_probs,  axis=0)   # (N, 14)
    labels_arr = np.concatenate(all_labels, axis=0)   # (N, 14)
    return probs_arr, labels_arr


# ══════════════════════════════════════════════════════════════════════════════
# 3. AUC-ROC EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_aucs(
    probs: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """Compute per-class AUC-ROC; returns NaN for any class with no positives."""
    aucs = {}
    for i, label in enumerate(DISEASE_LABELS):
        n_pos = labels[:, i].sum()
        if n_pos == 0:
            print(f"  Warning: '{label}' has no positive examples in test set — skipping.")
            aucs[label] = float("nan")
        else:
            aucs[label] = roc_auc_score(labels[:, i], probs[:, i])
    return aucs


def print_auc_table(aucs: dict[str, float]) -> float:
    """Print a per-class AUC table sorted descending and return mean AUC."""
    print("\n" + "═" * 45)
    print(f"  {'Disease':<22}  {'AUC-ROC':>8}  {'Positives':>9}")
    print("─" * 45)

    valid_aucs = []
    # Sort descending by AUC so best-performing classes are at the top
    for label, auc in sorted(aucs.items(), key=lambda x: x[1], reverse=True):
        if not np.isnan(auc):
            valid_aucs.append(auc)
        auc_str = f"{auc:.4f}" if not np.isnan(auc) else "  N/A  "
        print(f"  {label:<22}  {auc_str:>8}")

    mean_auc = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
    print("─" * 45)
    print(f"  {'Mean AUC':<22}  {mean_auc:.4f}")
    print("═" * 45 + "\n")
    return mean_auc


# ══════════════════════════════════════════════════════════════════════════════
# 4. ROC CURVE FIGURE
# ══════════════════════════════════════════════════════════════════════════════

def plot_roc_curves(
    probs: np.ndarray,
    labels: np.ndarray,
    aucs: dict[str, float],
    out_path: Path,
) -> None:
    """Plot all 14 ROC curves on a 3×5 grid and save to out_path."""
    n_cols  = 5
    n_rows  = 3   # 15 panels — one will be empty for the 14th class
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3.2))
    axes = axes.flatten()

    for i, label in enumerate(DISEASE_LABELS):
        ax  = axes[i]
        auc = aucs[label]

        if not np.isnan(auc):
            fpr, tpr, _ = roc_curve(labels[:, i], probs[:, i])
            ax.plot(fpr, tpr, color="steelblue", lw=1.8, label=f"AUC={auc:.3f}")
        else:
            ax.text(0.5, 0.5, "No positives", ha="center", va="center",
                    transform=ax.transAxes, color="grey")

        # Diagonal = random classifier baseline
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.set_title(f"{label}\nAUC={auc:.3f}" if not np.isnan(auc) else label,
                     fontsize=8.5)
        ax.set_xlabel("FPR", fontsize=7)
        ax.set_ylabel("TPR", fontsize=7)
        ax.tick_params(labelsize=6)

    # Hide the unused 15th panel
    axes[14].axis("off")

    fig.suptitle("NIH CXR-14 — Per-class ROC Curves", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ROC curves → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. GRAD-CAM
# ══════════════════════════════════════════════════════════════════════════════

class GradCAM:
    """
    Grad-CAM for DenseNet-121.

    Hooks into features.denseblock4 (7×7×1024) rather than features.norm5
    because norm5 has an inplace relu_ that conflicts with backward hooks.
    """

    def __init__(self, model: nn.Module, target_layer_name: str = "features.denseblock4"):
        self.model  = model
        self.device = next(model.parameters()).device

        # These will be filled by the hooks during the forward/backward pass
        self._activations: torch.Tensor | None = None
        self._gradients:   torch.Tensor | None = None

        # Register hooks on the target layer
        target_layer = dict(model.named_modules())[target_layer_name]

        # Forward hook: captures the output tensor of the layer
        self._fwd_handle = target_layer.register_forward_hook(self._save_activations)

        # Backward hook: captures the gradient of the loss w.r.t. the layer output
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        # Clone before detaching: denseblock4's output tensor may be reused
        # inplace by the next layer (norm5 → relu_).  Cloning gives us our
        # own copy that cannot be modified by subsequent inplace operations.
        self._activations = output.clone().detach()

    def _save_gradients(self, module, grad_input, grad_output):
        # grad_output[0] is the gradient flowing back into the layer output
        self._gradients = grad_output[0].detach()

    def remove_hooks(self):
        """Call this when done to free memory."""
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def generate(
        self,
        image_tensor: torch.Tensor,   # (1, 3, 224, 224)
        target_class: int | None = None,
    ) -> tuple[np.ndarray, int, float]:
        """
        Return (cam, target_class, prob). Defaults to the top-predicted class.
        cam is a float32 (H, W) array normalised to [0, 1].
        """
        self.model.eval()
        # image_tensor needs grad so the computation graph is built through it
        image_tensor = image_tensor.requires_grad_(False)

        # ── forward pass ──────────────────────────────────────────────────────
        # The forward hook fires here, saving activations at features.norm5
        logits = self.model(image_tensor)          # (1, 14)
        probs  = torch.sigmoid(logits)             # (1, 14)

        if target_class is None:
            target_class = int(probs.argmax(dim=1).item())

        prob = float(probs[0, target_class].item())

        self.model.zero_grad()
        logits[0, target_class].backward()

        A     = self._activations[0]   # (1024, 7, 7)
        grads = self._gradients[0]     # (1024, 7, 7)

        alpha = grads.mean(dim=(1, 2))   # per-channel importance weights
        cam   = torch.einsum("k, k h w -> h w", alpha, A)
        cam   = torch.relu(cam)

        # Upsample from 7×7 to the original input size (224×224)
        cam = cam.unsqueeze(0).unsqueeze(0)   # (1, 1, 7, 7) for interpolate
        cam = torch.nn.functional.interpolate(
            cam,
            size=(image_tensor.shape[2], image_tensor.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()     # (224, 224)

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, target_class, prob


def _denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Reverse ImageNet normalisation and return a uint8 numpy array (H, W, 3).

    We need the original pixel values to make the X-ray visible underneath
    the heatmap.
    """
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.cpu().numpy().transpose(1, 2, 0)   # (H, W, 3)
    img  = img * std + mean
    img  = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def save_gradcam_figure(
    original_img:    np.ndarray,    # (H, W, 3) uint8, denormalised
    cam:             np.ndarray,    # (H, W)    float32 in [0,1]
    image_filename:  str,
    target_class:    int,
    prob:            float,
    true_labels:     np.ndarray,    # (14,) binary
    out_path:        Path,
) -> None:
    """
    Save a three-panel figure:
      Left   — original X-ray (grayscale appearance)
      Centre — the raw Grad-CAM heatmap
      Right  — heatmap blended over the X-ray

    The blend uses the 'jet' colormap on the CAM (red = most activated)
    with alpha=0.45 so the underlying anatomy is still visible.
    """
    # Convert CAM to a coloured RGB image using the 'jet' colormap
    colormap    = cm.get_cmap("jet")
    cam_rgb     = (colormap(cam)[:, :, :3] * 255).astype(np.uint8)  # (H, W, 3)

    # Blend: 55% original + 45% heatmap
    blend = (0.55 * original_img + 0.45 * cam_rgb).astype(np.uint8)

    # Determine the ground-truth and predicted labels for the title
    predicted_name = DISEASE_LABELS[target_class]
    true_names = [DISEASE_LABELS[i] for i, v in enumerate(true_labels) if v == 1]
    true_str   = ", ".join(true_names) if true_names else "No Finding"

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(original_img, cmap="bone")
    axes[0].set_title("Original X-ray", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Grad-CAM", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(blend)
    axes[2].set_title(
        f"Overlay\nPredicted: {predicted_name} ({prob:.2f})\nTrue: {true_str}",
        fontsize=8.5,
    )
    axes[2].axis("off")

    fig.suptitle(image_filename, fontsize=9, color="grey")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_gradcam(
    model:       nn.Module,
    test_ds,
    image_index: dict,
    device:      torch.device,
    n_samples:   int = 5,
    seed:        int = 42,
) -> None:
    """
    Pick n_samples random test images, run Grad-CAM, and save figures.

    We prefer images that have at least one positive label so the heatmap
    shows where a real finding was detected (or missed).  If fewer than
    n_samples positive images exist we fall back to random selection.
    """
    rng = np.random.default_rng(seed)
    df  = test_ds.df

    # Prefer images with at least one disease label
    from dataset import DISEASE_LABELS as DL
    has_finding = df[DL].sum(axis=1) > 0
    candidate_df = df[has_finding] if has_finding.sum() >= n_samples else df

    sample_rows = candidate_df.sample(n=n_samples, random_state=int(seed))

    transform  = get_val_transforms()
    grad_cam   = GradCAM(model, target_layer_name="features.denseblock4")

    for i, (_, row) in enumerate(sample_rows.iterrows()):
        fname    = row["Image Index"]
        img_path = image_index[fname]

        # Load and preprocess the image exactly as during training
        pil_img  = Image.open(img_path).convert("RGB")
        tensor   = transform(pil_img).unsqueeze(0).to(device)   # (1,3,224,224)

        # Generate the Grad-CAM heatmap for the most-predicted class
        cam, target_cls, prob = grad_cam.generate(tensor, target_class=None)

        # Denormalise for display
        original_np = _denormalize(tensor.squeeze(0))

        true_labels = row[DISEASE_LABELS].values.astype(np.float32)

        out_path = FIGURES_DIR / f"gradcam_{i+1}_{fname.replace('.png', '')}.png"
        save_gradcam_figure(
            original_np, cam, fname, target_cls, prob, true_labels, out_path
        )
        print(
            f"  Grad-CAM {i+1}/{n_samples}: {fname}  "
            f"→ {DISEASE_LABELS[target_cls]} (p={prob:.3f})  saved"
        )

    grad_cam.remove_hooks()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(ckpt_path: Path, n_gradcam: int = 5) -> None:
    device = get_device()
    print(f"Device: {device}\n")

    # 1. Load model
    model = load_model(ckpt_path, device)

    # 2. Build dataloaders — we only need the test split
    print("Building DataLoaders …")
    _, _, test_loader, _ = build_dataloaders(num_workers=4)
    test_ds = test_loader.dataset

    # 3. Collect predictions
    print("Running inference on test set …")
    probs, labels = collect_predictions(model, test_loader, device)
    print(f"Collected predictions: probs {probs.shape}, labels {labels.shape}")

    # 4. Compute and print AUC table
    aucs = compute_aucs(probs, labels)
    mean_auc = print_auc_table(aucs)
    print(f"Mean AUC-ROC: {mean_auc:.4f}\n")

    # 5. Plot ROC curves
    plot_roc_curves(probs, labels, aucs, FIGURES_DIR / "roc_curves.png")

    # 6. Grad-CAM
    if n_gradcam > 0:
        print(f"\nGenerating {n_gradcam} Grad-CAM visualisations …")
        image_index = build_image_index(RAW_DIR)
        run_gradcam(model, test_ds, image_index, device,
                    n_samples=n_gradcam)

    print("\nEvaluation complete.")
    print(f"Figures saved to: {FIGURES_DIR}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate chest X-ray classifier")
    p.add_argument(
        "--ckpt", type=Path, default=DEFAULT_CKPT,
        help="path to checkpoint (default: outputs/checkpoints/best_model.pth)",
    )
    p.add_argument(
        "--n-gradcam", type=int, default=5,
        help="number of Grad-CAM figures to generate (0 to skip)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(ckpt_path=args.ckpt, n_gradcam=args.n_gradcam)
