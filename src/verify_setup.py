"""
verify_setup.py
---------------
End-to-end sanity check AFTER the dataset is downloaded.

  1. Checks all expected files exist in data/raw/
  2. Runs the DataLoader smoke test (one batch, no GPU needed)
  3. Reports the chosen device (MPS / CPU)

Run with:  python src/verify_setup.py
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

EXPECTED_FILES = [
    "Data_Entry_2017.csv",
    "train_val_list.txt",
    "test_list.txt",
    "BBox_List_2017.csv",
]

def check_files():
    print("=== Checking expected files ===")
    ok = True
    for fname in EXPECTED_FILES:
        path = RAW_DIR / fname
        if path.exists():
            print(f"  [OK] {fname}")
        else:
            print(f"  [MISSING] {fname}")
            ok = False

    # At least one images_* folder
    img_dirs = list(RAW_DIR.glob("images_*/images"))
    if img_dirs:
        n_images = sum(1 for d in img_dirs for _ in d.iterdir())
        print(f"  [OK] {len(img_dirs)} image folder(s) — {n_images:,} images total")
    else:
        print("  [MISSING] No images_*/images/ folders found")
        ok = False

    return ok

def main():
    all_ok = check_files()
    if not all_ok:
        print("\nSome files are missing. Run `python src/download_data.py` first.")
        sys.exit(1)

    print("\n=== DataLoader smoke test ===")
    # Import here so missing files don't block the import
    from dataset import build_dataloaders
    import torch

    train_loader, val_loader, test_loader, pos_weights = build_dataloaders(
        batch_size=8, num_workers=0
    )
    images, labels = next(iter(train_loader))
    print(f"  Batch images : {images.shape}  {images.dtype}")
    print(f"  Batch labels : {labels.shape}  {labels.dtype}")
    print(f"  Label range  : [{labels.min().item()}, {labels.max().item()}]")
    print(f"  pos_weights  : {pos_weights.tolist()}")

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"\n  Device: {device}")
    _ = images.to(device)
    print("  Tensor moved to device  OK")
    print("\nAll checks passed.")

if __name__ == "__main__":
    main()
