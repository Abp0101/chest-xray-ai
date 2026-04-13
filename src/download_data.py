"""
download_data.py
----------------
Downloads the NIH Chest X-Ray 14 dataset from Kaggle and unpacks it into
data/raw/.

The dataset is published on Kaggle as:
    nih-chest-xrays/data
    https://www.kaggle.com/datasets/nih-chest-xrays/data

It contains:
  - ~112,000 frontal-view X-ray images (PNG, 1024×1024)
  - Data_Entry_2017.csv  — per-image metadata + multi-label disease strings
  - train_val_list.txt / test_list.txt — official train/test splits
  - BBox_List_2017.csv  — bounding boxes for a subset of images

Usage:
    python src/download_data.py
"""

import os
import subprocess
import zipfile
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

KAGGLE_DATASET = "nih-chest-xrays/data"


def download():
    print(f"Downloading '{KAGGLE_DATASET}' to {RAW_DIR} …")
    # kaggle datasets download writes a zip next to --path
    subprocess.run(
        [
            "kaggle", "datasets", "download",
            KAGGLE_DATASET,
            "--path", str(RAW_DIR),
            "--unzip",          # unzip in-place; kaggle CLI removes the zip
        ],
        check=True,
    )
    print("Download complete.")
    # List what we got so the caller can verify
    entries = sorted(RAW_DIR.iterdir())
    print(f"\nContents of {RAW_DIR}:")
    for e in entries:
        size_mb = e.stat().st_size / 1e6 if e.is_file() else 0
        tag = f"  ({size_mb:.1f} MB)" if e.is_file() else "/"
        print(f"  {e.name}{tag}")


if __name__ == "__main__":
    download()
