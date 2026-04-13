"""
explore_data.py
---------------
Loads Data_Entry_2017.csv and produces:

  1. Class distribution bar chart  →  outputs/figures/class_distribution.png
  2. Missing-data report           →  printed to stdout
  3. Sample image grid             →  outputs/figures/sample_images.png
  4. Patient / image count summary →  printed to stdout

Run after download_data.py:
    python src/explore_data.py
"""

import ast
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# ── paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
FIGURES_DIR  = PROJECT_ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = RAW_DIR / "Data_Entry_2017.csv"

# The NIH dataset stores images in nested sub-folders:
#   images_001/images/, images_002/images/, … images_012/images/
# This helper finds a filename regardless of which sub-folder it lives in.
def find_image_dirs(raw_dir: Path) -> list[Path]:
    """Return a list of all …/images/ subdirectories under raw_dir."""
    return sorted(raw_dir.glob("images_*/images"))


def build_image_index(raw_dir: Path) -> dict[str, Path]:
    """
    Walk every images_*/images/ folder and build a filename→full-path map.
    This is O(n_files) once at startup; much faster than repeated glob searches.
    """
    index = {}
    for img_dir in find_image_dirs(raw_dir):
        for p in img_dir.iterdir():
            index[p.name] = p
    return index


# ── 1. load metadata ──────────────────────────────────────────────────────────

def load_metadata(csv_path: Path) -> pd.DataFrame:
    """
    Read Data_Entry_2017.csv.

    Key columns:
      Image Index        – filename, e.g. "00000001_000.png"
      Finding Labels     – pipe-separated disease names, e.g. "Atelectasis|Effusion"
                           "No Finding" means healthy
      Patient ID         – integer; one patient can have many images
      Patient Age        – age in years (some outliers > 100 present in raw data)
      Patient Gender     – M / F
      View Position      – PA (posteroanterior) or AP (anteroposterior)
      OriginalImage[Width,Height]  – native pixel dimensions
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} rows from {csv_path.name}")
    print(f"Columns: {list(df.columns)}\n")
    return df


# ── 2. parse multi-labels ─────────────────────────────────────────────────────

# The 14 pathology classes defined by NIH (plus the healthy pseudo-label)
DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]
ALL_LABELS = DISEASE_LABELS + ["No Finding"]


def expand_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add one binary column per label.
    'Atelectasis|Effusion' → Atelectasis=1, Effusion=1, everything else=0.

    This converts the problem from a single multi-class label into a
    multi-label binary classification (each image can have multiple diseases).
    """
    for label in ALL_LABELS:
        # str.contains works on pipe-delimited strings correctly because
        # each label name is unique and does not appear as a substring of another.
        df[label] = df["Finding Labels"].str.contains(label).astype(int)
    return df


# ── 3. missing data report ────────────────────────────────────────────────────

def report_missing(df: pd.DataFrame) -> None:
    print("=== Missing-data report ===")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("  No missing values in any column.\n")
    else:
        print(missing.to_string())
        print()

    # Age sanity check – values > 120 are data entry artifacts
    bad_age = (df["Patient Age"] > 120).sum()
    if bad_age:
        print(f"  Warning: {bad_age} rows have Patient Age > 120 (likely encoding errors).\n")

    # Check that every declared image file exists on disk
    image_index = build_image_index(RAW_DIR)
    n_total = len(df)
    n_found = df["Image Index"].isin(image_index).sum()
    print(f"  Image files found on disk: {n_found:,} / {n_total:,}")
    if n_found < n_total:
        missing_files = df.loc[~df["Image Index"].isin(image_index), "Image Index"]
        print(f"  First 5 missing: {missing_files.head().tolist()}")
    print()


# ── 4. class distribution plot ────────────────────────────────────────────────

def plot_class_distribution(df: pd.DataFrame, out_path: Path) -> None:
    """
    Bar chart showing how many images contain each label.

    Medical imaging datasets are almost always highly imbalanced:
    'No Finding' dominates, rare diseases have far fewer examples.
    This chart makes that imbalance visible so we can decide how to handle it
    (weighted sampling, focal loss, oversampling, etc.).
    """
    counts = {label: df[label].sum() for label in ALL_LABELS}
    counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(counts.keys(), counts.values(), color="steelblue", edgecolor="white")
    ax.set_title("NIH Chest X-Ray 14 — label frequency", fontsize=14, pad=12)
    ax.set_ylabel("Number of images")
    ax.set_xlabel("Label")
    plt.xticks(rotation=40, ha="right")

    # Annotate bar heights
    for bar, (label, count) in zip(bars, counts.items()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 200,
            f"{count:,}",
            ha="center", va="bottom", fontsize=7,
        )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved class distribution chart → {out_path}")


# ── 5. sample image grid ──────────────────────────────────────────────────────

def plot_sample_images(df: pd.DataFrame, image_index: dict, out_path: Path,
                       n_cols: int = 5, n_rows: int = 3) -> None:
    """
    Pick random images (one per row, varied labels) and show them in a grid.

    Each panel title is the short label string so you can see what a
    normal-looking 'Cardiomegaly' or 'Pneumothorax' X-ray looks like.
    """
    sample_df = df[df["Image Index"].isin(image_index)].sample(
        n=n_cols * n_rows, random_state=42
    )

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes = axes.flatten()

    for ax, (_, row) in zip(axes, sample_df.iterrows()):
        img_path = image_index[row["Image Index"]]
        img = Image.open(img_path).convert("L")   # grayscale
        ax.imshow(img, cmap="bone")               # bone colormap looks like X-ray film
        # Truncate long label strings to keep titles readable
        label = row["Finding Labels"]
        title = (label[:28] + "…") if len(label) > 30 else label
        ax.set_title(title, fontsize=7)
        ax.axis("off")

    fig.suptitle("NIH Chest X-Ray 14 — random sample", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved sample image grid   → {out_path}")


# ── 6. summary stats ──────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    print("=== Dataset summary ===")
    print(f"  Total images      : {len(df):,}")
    print(f"  Unique patients   : {df['Patient ID'].nunique():,}")
    print(f"  Avg images/patient: {len(df) / df['Patient ID'].nunique():.1f}")
    print(f"  View positions    : {df['View Position'].value_counts().to_dict()}")
    print(f"  Gender split      : {df['Patient Gender'].value_counts().to_dict()}")
    age_col = df["Patient Age"]
    print(f"  Age range         : {age_col[age_col <= 120].min()}–{age_col[age_col <= 120].max()} years")
    print()

    print("  Label counts (sorted):")
    for label in ALL_LABELS:
        n = int(df[label].sum())
        pct = 100 * n / len(df)
        print(f"    {label:<22} {n:>6,}  ({pct:4.1f}%)")
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    df = load_metadata(CSV_PATH)
    df = expand_labels(df)

    report_missing(df)
    print_summary(df)

    plot_class_distribution(df, FIGURES_DIR / "class_distribution.png")

    image_index = build_image_index(RAW_DIR)
    if image_index:
        plot_sample_images(df, image_index, FIGURES_DIR / "sample_images.png")
    else:
        print("No images found under data/raw – run download_data.py first.")


if __name__ == "__main__":
    main()
