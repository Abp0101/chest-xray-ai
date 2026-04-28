"""
DenseNet-121 with a 14-class multi-label head for NIH CXR-14.
No sigmoid in the forward pass — BCEWithLogitsLoss handles it.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import DenseNet121_Weights

from dataset import NUM_CLASSES   # 14


def build_model(num_classes: int = NUM_CLASSES, dropout: float = 0.5) -> nn.Module:
    """Load ImageNet-pretrained DenseNet-121 and replace the classifier head."""
    model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)

    in_features = model.classifier.in_features   # 1024
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )

    return model


# ── fine-tuning helpers ───────────────────────────────────────────────────────

def freeze_backbone(model: nn.Module) -> None:
    """Freeze all backbone layers; only model.classifier stays trainable."""
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[freeze_backbone] trainable params: {trainable:,} / {total:,}")


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze all parameters for full fine-tuning (stage 2)."""
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
