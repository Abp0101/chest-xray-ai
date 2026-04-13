"""
model.py
--------
Wraps torchvision's pretrained DenseNet-121 for 14-class multi-label
chest X-ray classification.

Why DenseNet-121?
-----------------
It is the architecture used in the original NIH paper (CheXNet, Rajpurkar
et al. 2017). Dense connections mean each layer receives feature maps from
all preceding layers, which helps the gradient flow to early layers and
allows the network to combine low-level texture with high-level structure —
both useful for reading X-rays.

Head replacement strategy
--------------------------
The ImageNet classifier (Linear 1024→1000 + implicit softmax) is replaced
with a single Linear 1024→14.  There is NO sigmoid here because
BCEWithLogitsLoss fuses sigmoid + BCE in a single, numerically stable
operation (avoids floating-point saturation at extreme logit values).

Two-stage fine-tuning
---------------------
Training the entire network from scratch on 100 k images is slow and risks
overwriting useful low-level features (edges, textures) learned from
ImageNet.  Instead:

  Stage 1 — head-only warm-up
    Freeze the DenseNet backbone.  Train only the new Linear(1024→14) for
    a few epochs.  This brings the random classifier weights close to the
    right scale before the backbone gradients are switched on.

  Stage 2 — full fine-tune
    Unfreeze everything.  Use a lower learning rate so the pretrained
    features adjust gradually rather than being destroyed.

freeze_backbone() / unfreeze_backbone() implement this pattern and are
called by train.py at the appropriate epoch.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import DenseNet121_Weights

from dataset import NUM_CLASSES   # 14


def build_model(num_classes: int = NUM_CLASSES, dropout: float = 0.5) -> nn.Module:
    """
    Load ImageNet-pretrained DenseNet-121 and replace its classifier head.

    Args:
        num_classes : number of output logits (14 for NIH CXR-14).
        dropout     : dropout probability applied before the final linear
                      layer.  Regularises the head during stage-1 warm-up
                      when only the classifier is trainable.

    Returns:
        model : nn.Module with the new 14-class head.
                All backbone parameters are initially TRAINABLE — call
                freeze_backbone() before stage-1 training.
    """
    # Load weights pretrained on ImageNet-1K (best publicly available
    # torchvision checkpoint for DenseNet-121).
    model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)

    # ── replace the classifier head ───────────────────────────────────────────
    # model.classifier is a single Linear(1024, 1000).
    # We keep the same 1024-d input but output num_classes logits.
    # A Dropout before the linear layer reduces overfitting on the head
    # during stage 1, when only this part of the network is updated.
    in_features = model.classifier.in_features   # 1024
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )

    return model


# ── fine-tuning helpers ───────────────────────────────────────────────────────

def freeze_backbone(model: nn.Module) -> None:
    """
    Freeze all DenseNet feature layers; leave model.classifier trainable.

    Call this before stage-1 (head-only) training.
    Setting requires_grad=False prevents gradient computation and parameter
    updates for the backbone, so only the new head learns.
    """
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[freeze_backbone] trainable params: {trainable:,} / {total:,}")


def unfreeze_backbone(model: nn.Module) -> None:
    """
    Unfreeze the entire network for end-to-end fine-tuning (stage 2).

    Call this once the head has warmed up.  Use a lower LR in the optimiser
    (or a per-layer LR schedule) to avoid overwriting pretrained features.
    """
    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[unfreeze_backbone] trainable params: {trainable:,}")


def count_parameters(model: nn.Module) -> dict:
    """Return a dict with total / trainable / frozen parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


# ── smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}\n")

    model = build_model()
    print("=== Before freezing ===")
    print(count_parameters(model))

    freeze_backbone(model)
    print("\n=== After freeze_backbone() ===")
    print(count_parameters(model))

    unfreeze_backbone(model)
    print("\n=== After unfreeze_backbone() ===")
    print(count_parameters(model))

    # Forward pass sanity check
    model = model.to(device)
    dummy = torch.randn(4, 3, 224, 224, device=device)
    with torch.no_grad():
        out = model(dummy)
    print(f"\nForward pass: input {dummy.shape} → output {out.shape}")
    print("Logit range (untrained head):", out.min().item(), "→", out.max().item())
