"""
Two-stage training pipeline for DenseNet-121 on NIH CXR-14.

Stage 1: backbone frozen, head-only warm-up.
Stage 2: full network fine-tuned at lower LR with ReduceLROnPlateau.

Usage:
    python src/train.py
    python src/train.py --no-wandb
    python src/train.py --epochs-stage1 5 --epochs-stage2 20 --batch 64
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add src/ to path so we can import sibling modules when running from the
# project root or from inside src/.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import build_dataloaders, DISEASE_LABELS
from model import build_model, freeze_backbone, unfreeze_backbone

# W&B is optional — if not installed or not wanted, every WandbLogger method
# becomes a silent no-op so the rest of the code is unchanged.
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# ── paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
CKPT_DIR      = PROJECT_ROOT / "outputs" / "checkpoints"
FIGURES_DIR   = PROJECT_ROOT / "outputs" / "figures"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
LOSS_LOG_PATH = PROJECT_ROOT / "outputs" / "loss_log.csv"

from dataset import RAW_DIR   # needed for Grad-CAM image lookup in W&B block


# ── W&B logger ────────────────────────────────────────────────────────────────

class WandbLogger:
    """Thin W&B wrapper — all methods are no-ops when W&B is disabled or unavailable."""

    def __init__(
        self,
        enabled: bool,
        project: str,
        config:  dict,
    ) -> None:
        self.enabled = enabled and _WANDB_AVAILABLE

        if self.enabled:
            if not _WANDB_AVAILABLE:
                print("Warning: wandb not installed — logging disabled.")
                self.enabled = False
                return
            wandb.init(
                project=project,
                config=config,
                # Save code so you can reproduce runs from the W&B UI
                save_code=True,
            )
            print(f"W&B run: {wandb.run.url}\n")

    def log_epoch(
        self,
        epoch:      int,
        stage:      str,
        train_loss: float,
        val_loss:   float,
        lr:         float,
    ) -> None:
        """Log scalar metrics for one epoch."""
        if not self.enabled:
            return
        wandb.log({
            "epoch":      epoch,
            "stage":      stage,
            "train/loss": train_loss,
            "val/loss":   val_loss,
            "train/lr":   lr,
        }, step=epoch)

    def log_auc(self, aucs: dict[str, float], mean_auc: float) -> None:
        """
        Log per-class and mean AUC after the test set evaluation.
        Called once at the end of training.
        """
        if not self.enabled:
            return
        metrics = {"test/mean_auc": mean_auc}
        for label, auc in aucs.items():
            if not __import__("math").isnan(auc):
                metrics[f"test/auc_{label}"] = auc
        wandb.log(metrics)

    def log_roc_figure(self, fig_path: Path) -> None:
        """Upload the saved ROC curves PNG as a W&B Image."""
        if not self.enabled or not fig_path.exists():
            return
        wandb.log({"roc_curves": wandb.Image(str(fig_path),
                                             caption="Per-class ROC curves")})

    def log_gradcam_figures(self, figures_dir: Path, n: int = 3) -> None:
        """
        Upload the first n Grad-CAM overlay PNGs from figures_dir.
        Files are matched by the gradcam_* prefix written by evaluate.py.
        """
        if not self.enabled:
            return
        paths = sorted(figures_dir.glob("gradcam_*.png"))[:n]
        if not paths:
            return
        images = [wandb.Image(str(p), caption=p.stem) for p in paths]
        wandb.log({"gradcam_samples": images})

    def finish(self) -> None:
        if self.enabled:
            wandb.finish()


# ── device ────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """
    Prefer MPS (Apple Silicon GPU) → fall back to CPU.
    CUDA is not listed because this project targets Apple Silicon.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── one epoch of training ─────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """Run one training epoch and return the mean loss."""
    model.train()
    running_loss = 0.0
    n_batches = len(loader)

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Forward pass: DenseNet returns raw logits, shape (B, 14).
        # No sigmoid here — BCEWithLogitsLoss handles it.
        logits = model(images)
        loss   = criterion(logits, labels)

        loss.backward()

        # Clip gradients to L2 norm ≤ 1.0.
        # Without this, a few large batches during fine-tuning can destabilise
        # the backbone weights with one very large update step.
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()

        # Print a progress line every 100 batches so you can see training
        # is alive without flooding the terminal.
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == n_batches:
            avg = running_loss / (batch_idx + 1)
            print(
                f"  Epoch {epoch}  [{batch_idx+1:>4}/{n_batches}]  "
                f"train loss: {avg:.4f}",
                end="\r",
            )

    print()  # newline after the \r progress line
    return running_loss / n_batches


# ── validation pass ───────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Return average validation loss over the DataLoader."""
    model.eval()
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss   = criterion(logits, labels)
        running_loss += loss.item()

    return running_loss / len(loader)


# ── checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    path: Path,
) -> None:
    """Save model + optimiser state to a device-agnostic checkpoint."""
    torch.save(
        {
            "epoch":      epoch,
            "val_loss":   val_loss,
            "model_state_dict":     model.cpu().state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
    # Move back to the original device after saving
    model.to(get_device())


def load_checkpoint(model: nn.Module, path: Path, device: torch.device) -> dict:
    """Load a checkpoint saved by save_checkpoint(); return the metadata dict."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt


# ── loss logger ───────────────────────────────────────────────────────────────

class LossLogger:
    """
    Appends one row per epoch to a CSV file.
    Columns: epoch, stage, train_loss, val_loss, elapsed_sec

    Keeping losses in a CSV means you can plot them later with
    pandas + matplotlib without re-running training.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        # Write the header row (overwrite any previous log)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "stage", "train_loss", "val_loss", "elapsed_sec"])

    def log(self, epoch: int, stage: str, train_loss: float,
            val_loss: float, elapsed: float) -> None:
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, stage, f"{train_loss:.6f}",
                             f"{val_loss:.6f}", f"{elapsed:.1f}"])


# ── main training function ────────────────────────────────────────────────────

def train(
    epochs_stage1:  int   = 3,
    epochs_stage2:  int   = 12,
    batch_size:     int   = 32,
    lr_stage1:      float = 1e-3,
    lr_stage2:      float = 1e-4,
    num_workers:    int   = 4,
    dropout:        float = 0.5,
    seed:           int   = 42,
    use_wandb:      bool  = True,
    wandb_project:  str   = "chest-xray-classifier",
) -> None:
    """Run the full two-stage training loop and save checkpoints."""
    torch.manual_seed(seed)
    device = get_device()
    print(f"Training on device: {device}\n")

    wb = WandbLogger(
        enabled=use_wandb,
        project=wandb_project,
        config=dict(
            epochs_stage1=epochs_stage1,
            epochs_stage2=epochs_stage2,
            batch_size=batch_size,
            lr_stage1=lr_stage1,
            lr_stage2=lr_stage2,
            dropout=dropout,
            seed=seed,
            architecture="densenet121",
            dataset="NIH-CXR-14",
        ),
    )

    # ── data ──────────────────────────────────────────────────────────────────
    print("Building DataLoaders …")
    train_loader, val_loader, _, pos_weights = build_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )

    # ── model ─────────────────────────────────────────────────────────────────
    print("Building model …")
    model = build_model(dropout=dropout).to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weights.to(device)
    )

    # ── loss logger ───────────────────────────────────────────────────────────
    logger = LossLogger(LOSS_LOG_PATH)

    best_val_loss = float("inf")
    epoch_global  = 0    # tracks epoch number across both stages for logging

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 1 — head-only warm-up
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 60)
    print("STAGE 1 — head warm-up (backbone frozen)")
    print("═" * 60)

    freeze_backbone(model)

    optimizer_s1 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_stage1,
        weight_decay=1e-5,
    )

    for epoch in range(1, epochs_stage1 + 1):
        epoch_global += 1
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer_s1, device, epoch_global
        )
        val_loss = evaluate(model, val_loader, criterion, device)
        elapsed  = time.time() - t0

        current_lr = optimizer_s1.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch_global:>3} [stage1]  "
            f"train: {train_loss:.4f}  val: {val_loss:.4f}  "
            f"({elapsed:.0f}s)"
        )
        logger.log(epoch_global, "stage1", train_loss, val_loss, elapsed)
        wb.log_epoch(epoch_global, "stage1", train_loss, val_loss, current_lr)

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer_s1, epoch_global, val_loss,
                CKPT_DIR / "best_model.pth",
            )
            print(f"  ✓ New best val loss: {best_val_loss:.4f} — checkpoint saved")

    # Always save the last state so stage 2 can resume if interrupted
    save_checkpoint(
        model, optimizer_s1, epoch_global, val_loss,
        CKPT_DIR / "last_model.pth",
    )

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 2 — full fine-tune
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 60)
    print("STAGE 2 — full fine-tune (backbone unfrozen)")
    print("═" * 60)

    unfreeze_backbone(model)

    # Fresh optimiser resets stage-1 momentum before fine-tuning the backbone.
    optimizer_s2 = torch.optim.Adam(
        model.parameters(),
        lr=lr_stage2,
        weight_decay=1e-5,
    )

    # Reduce LR by factor 0.5 if val loss does not improve for 2 epochs.
    # 'min' mode means we reduce when the monitored quantity stops decreasing.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_s2,
        mode="min",
        factor=0.5,
        patience=2,
        
    )

    for epoch in range(1, epochs_stage2 + 1):
        epoch_global += 1
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer_s2, device, epoch_global
        )
        val_loss = evaluate(model, val_loader, criterion, device)
        elapsed  = time.time() - t0

        # Feed val loss to the scheduler so it can decide whether to reduce LR
        scheduler.step(val_loss)

        current_lr = optimizer_s2.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch_global:>3} [stage2]  "
            f"train: {train_loss:.4f}  val: {val_loss:.4f}  "
            f"lr: {current_lr:.2e}  ({elapsed:.0f}s)"
        )
        logger.log(epoch_global, "stage2", train_loss, val_loss, elapsed)
        wb.log_epoch(epoch_global, "stage2", train_loss, val_loss, current_lr)

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer_s2, epoch_global, val_loss,
                CKPT_DIR / "best_model.pth",
            )
            print(f"  ✓ New best val loss: {best_val_loss:.4f} — checkpoint saved")

        # Always overwrite last_model.pth
        save_checkpoint(
            model, optimizer_s2, epoch_global, val_loss,
            CKPT_DIR / "last_model.pth",
        )

    print(f"\nTraining complete.  Best val loss: {best_val_loss:.4f}")
    print(f"Best checkpoint : {CKPT_DIR / 'best_model.pth'}")
    print(f"Loss log        : {LOSS_LOG_PATH}")

    if wb.enabled:
        print("\nRunning post-training evaluation for W&B …")
        from evaluate import (
            collect_predictions, compute_aucs, print_auc_table,
            plot_roc_curves, run_gradcam, build_image_index,
        )

        # Reload best weights (training finished on last weights, not best)
        best_model = build_model(dropout=dropout).to(device)
        best_ckpt  = torch.load(
            CKPT_DIR / "best_model.pth", map_location=device, weights_only=True
        )
        best_model.load_state_dict(best_ckpt["model_state_dict"])
        best_model.eval()

        # AUC on test set
        _, _, test_loader, _ = build_dataloaders(
            batch_size=batch_size, num_workers=num_workers, seed=seed
        )
        probs, labels = collect_predictions(best_model, test_loader, device)
        aucs     = compute_aucs(probs, labels)
        mean_auc = print_auc_table(aucs)
        wb.log_auc(aucs, mean_auc)

        # ROC curves figure
        roc_path = FIGURES_DIR / "roc_curves.png"
        plot_roc_curves(probs, labels, aucs, roc_path)
        wb.log_roc_figure(roc_path)

        # 3 Grad-CAM samples
        image_index = build_image_index(RAW_DIR)
        run_gradcam(best_model, test_loader.dataset, image_index,
                    device, n_samples=3)
        wb.log_gradcam_figures(FIGURES_DIR, n=3)

    wb.finish()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DenseNet-121 on NIH CXR-14")
    p.add_argument("--epochs-stage1", type=int,   default=3,    help="head warm-up epochs")
    p.add_argument("--epochs-stage2", type=int,   default=12,   help="full fine-tune epochs")
    p.add_argument("--batch",         type=int,   default=32,   help="batch size")
    p.add_argument("--lr-stage1",     type=float, default=1e-3, help="LR for stage 1")
    p.add_argument("--lr-stage2",     type=float, default=1e-4, help="LR for stage 2")
    p.add_argument("--workers",       type=int,   default=4,    help="DataLoader workers")
    p.add_argument("--dropout",       type=float, default=0.5,  help="head dropout rate")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--no-wandb",      action="store_true",
                   help="disable Weights & Biases logging")
    p.add_argument("--wandb-project", type=str, default="chest-xray-classifier",
                   help="W&B project name")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        epochs_stage1=args.epochs_stage1,
        epochs_stage2=args.epochs_stage2,
        batch_size=args.batch,
        lr_stage1=args.lr_stage1,
        lr_stage2=args.lr_stage2,
        num_workers=args.workers,
        dropout=args.dropout,
        seed=args.seed,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
    )
