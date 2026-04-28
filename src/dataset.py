"""
Dataset and DataLoader setup for NIH CXR-14.

Uses the official train_val_list.txt / test_list.txt splits with a
patient-level 90/10 train/val split. No horizontal flips (heart position),
conservative crop (≥90%) to preserve clinically relevant regions.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

# ── paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

CSV_PATH        = RAW_DIR / "Data_Entry_2017.csv"
TRAIN_VAL_LIST  = RAW_DIR / "train_val_list.txt"
TEST_LIST       = RAW_DIR / "test_list.txt"

# ── label definitions ─────────────────────────────────────────────────────────
DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]
NUM_CLASSES = len(DISEASE_LABELS)   # 14

# ImageNet statistics — used because we fine-tune pretrained ResNet/DenseNet.
# Each channel gets the same value since our images are grayscale converted
# to 3-channel RGB (by repeating the single channel three times).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

IMAGE_SIZE = 224   # standard input size for most pretrained models


# ── transforms ────────────────────────────────────────────────────────────────

def get_train_transforms() -> transforms.Compose:
    """Conservative augmentation — small crop, gentle rotation, no horizontal flip."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.9, 1.0)),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms() -> transforms.Compose:
    """Deterministic centre-crop pipeline for val/test."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ── dataset ───────────────────────────────────────────────────────────────────

class ChestXrayDataset(Dataset):
    """
    Returns (image_tensor, label_tensor) pairs for the NIH CXR-14 dataset.
    image: FloatTensor (3, H, W), labels: FloatTensor (14,) multi-hot.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_index: dict,
        transform: transforms.Compose,
    ):
        # Keep only rows whose image file actually exists on disk
        self.df = df[df["Image Index"].isin(image_index)].reset_index(drop=True)
        self.image_index = image_index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # ── load image ────────────────────────────────────────────────────────
        img_path = self.image_index[row["Image Index"]]
        # Open as grayscale first, then convert to RGB.
        # Converting L→RGB simply repeats the single channel three times;
        # it does NOT add color information but satisfies (3, H, W) convention.
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)   # → (3, IMAGE_SIZE, IMAGE_SIZE) float32

        # ── build label vector ────────────────────────────────────────────────
        # Multi-hot float32 tensor, shape (14,).
        # A value of 1.0 means that pathology is present in this image.
        labels = torch.tensor(
            row[DISEASE_LABELS].values.astype(np.float32),
            dtype=torch.float32,
        )

        return image, labels


# ── helper functions ──────────────────────────────────────────────────────────

def _load_split_list(path: Path) -> set[str]:
    """Read one filename per line; return a set for O(1) membership tests."""
    return set(path.read_text().strip().splitlines())


def _parse_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add a binary column for each of the 14 disease labels."""
    for label in DISEASE_LABELS:
        df[label] = df["Finding Labels"].str.contains(label, regex=False).astype(int)
    return df


def build_image_index(raw_dir: Path) -> dict[str, Path]:
    """
    Scan all images_*/images/ subdirectories and return a filename→Path map.
    Called once; avoids repeated filesystem searches during training.
    """
    index: dict[str, Path] = {}
    for img_dir in sorted(raw_dir.glob("images_*/images")):
        for p in img_dir.iterdir():
            index[p.name] = p
    return index


def compute_pos_weights(train_df: pd.DataFrame) -> torch.Tensor:
    """
    Compute per-class positive weights for BCEWithLogitsLoss.

    For an imbalanced multi-label problem, pos_weight = neg_count / pos_count
    tells the loss function to penalise false negatives more heavily for rare
    classes, countering the imbalance without changing the data.

    Returns a FloatTensor of shape (14,).
    """
    pos = train_df[DISEASE_LABELS].sum(axis=0).values.astype(np.float32)
    neg = len(train_df) - pos
    # Clip denominator to avoid division by zero for any label with 0 positives
    weights = neg / np.maximum(pos, 1.0)
    return torch.tensor(weights, dtype=torch.float32)


# ── public API ────────────────────────────────────────────────────────────────

def build_dataloaders(
    raw_dir: Path = RAW_DIR,
    batch_size: int = 32,
    num_workers: int = 4,
    val_fraction: float = 0.1,
    use_weighted_sampler: bool = True,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Return (train_loader, val_loader, test_loader, pos_weights).

    Follows the official NIH train_val/test split files with a patient-level
    90/10 train/val split. WeightedRandomSampler up-samples rare-disease images.
    """
    rng = np.random.default_rng(seed)

    # ── load and annotate metadata ────────────────────────────────────────────
    df = pd.read_csv(raw_dir / "Data_Entry_2017.csv")
    df = _parse_labels(df)

    # ── apply official train / test split ─────────────────────────────────────
    train_val_files = _load_split_list(raw_dir / "train_val_list.txt")
    test_files      = _load_split_list(raw_dir / "test_list.txt")

    train_val_df = df[df["Image Index"].isin(train_val_files)].copy()
    test_df      = df[df["Image Index"].isin(test_files)].copy()

    # ── patient-level train / val split ───────────────────────────────────────
    # Group by patient so no patient straddles train and val.
    # This is important: a patient can have 10+ images, and if some are in
    # train and some in val the model can memorise patient anatomy.
    patients = train_val_df["Patient ID"].unique()
    rng.shuffle(patients)
    n_val_patients = max(1, int(len(patients) * val_fraction))
    val_patients   = set(patients[:n_val_patients])
    train_patients = set(patients[n_val_patients:])

    train_df = train_val_df[train_val_df["Patient ID"].isin(train_patients)].copy()
    val_df   = train_val_df[train_val_df["Patient ID"].isin(val_patients)].copy()

    print(
        f"Split sizes — train: {len(train_df):,}  "
        f"val: {len(val_df):,}  "
        f"test: {len(test_df):,}"
    )

    # ── image index ───────────────────────────────────────────────────────────
    image_index = build_image_index(raw_dir)
    print(f"Images found on disk: {len(image_index):,}")

    # ── datasets ──────────────────────────────────────────────────────────────
    train_ds = ChestXrayDataset(train_df, image_index, get_train_transforms())
    val_ds   = ChestXrayDataset(val_df,   image_index, get_val_transforms())
    test_ds  = ChestXrayDataset(test_df,  image_index, get_val_transforms())

    # ── weighted sampler for training ─────────────────────────────────────────
    sampler = None
    if use_weighted_sampler:
        # Each image gets a weight = 1 / (number of positive labels it has).
        # Images with rare diseases therefore get sampled more often.
        # Images labelled "No Finding" (label_sum == 0) get weight 1.
        label_sums = train_ds.df[DISEASE_LABELS].sum(axis=1).values
        sample_weights = np.where(label_sums > 0, 1.0 / label_sums, 1.0)
        sample_weights = torch.tensor(sample_weights, dtype=torch.float64)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_ds),
            replacement=True,       # with-replacement so rare classes repeat
        )

    # ── data loaders ──────────────────────────────────────────────────────────
    # pin_memory=True speeds up CPU→GPU transfers by using page-locked memory.
    # On MPS (Apple Silicon) it has no effect but doesn't hurt.
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),   # mutually exclusive with sampler
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,              # drop the last incomplete batch for stable BN stats
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # ── pos_weights for loss function ─────────────────────────────────────────
    pos_weights = compute_pos_weights(train_ds.df)

    return train_loader, val_loader, test_loader, pos_weights


# ── smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Quick sanity check: build loaders and pull one batch.
    Run with:   python src/dataset.py
    """
    train_loader, val_loader, test_loader, pos_weights = build_dataloaders(
        batch_size=16, num_workers=0   # num_workers=0 avoids multiprocess issues when run directly
    )

    images, labels = next(iter(train_loader))
    print(f"\nBatch shapes — images: {images.shape}  labels: {labels.shape}")
    print(f"Image dtype / range  : {images.dtype}  [{images.min():.2f}, {images.max():.2f}]")
    print(f"Labels dtype         : {labels.dtype}")
    print(f"pos_weights shape    : {pos_weights.shape}")
    print(f"pos_weights          : {pos_weights.numpy().round(1)}")

    # Confirm MPS is accessible
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"\nDevice: {device}")
    images_gpu = images.to(device)
    print(f"Tensor on {device}: {images_gpu.shape}  OK")
