"""
fix_filenames.py
────────────────
Renames image + label pairs that have non-ASCII characters in their filenames.
Fixes Korean/CJK filenames from Hanshin University (ALL_image dataset) that
cause cv2.imread to fail on Windows.

Run BEFORE data_augmentation.py:
    python fix_filenames.py
"""

import re
import shutil
from pathlib import Path

def sanitize(name: str) -> str:
    """Replace non-ASCII characters with underscores, collapse repeats."""
    # Replace non-ASCII chars with underscore
    s = re.sub(r'[^\x00-\x7F]+', '_', name)
    # Collapse multiple underscores
    s = re.sub(r'_+', '_', s)
    return s.strip('_')


def fix_split(img_dir: Path, lbl_dir: Path) -> int:
    renamed = 0
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for img_path in list(img_dir.iterdir()):
        stem = img_path.stem
        suffix = img_path.suffix

        # Check if name has non-ASCII chars
        if all(ord(c) < 128 for c in img_path.name):
            continue

        new_stem = sanitize(stem)
        new_img = img_dir / (new_stem + suffix)
        new_lbl = lbl_dir / (new_stem + ".txt")
        old_lbl = lbl_dir / (stem + ".txt")

        # Avoid collision
        if new_img.exists() and new_img != img_path:
            idx = 1
            while (img_dir / f"{new_stem}_{idx}{suffix}").exists():
                idx += 1
            new_stem = f"{new_stem}_{idx}"
            new_img = img_dir / (new_stem + suffix)
            new_lbl = lbl_dir / (new_stem + ".txt")

        img_path.rename(new_img)
        if old_lbl.exists():
            old_lbl.rename(new_lbl)

        renamed += 1

    return renamed


def main(dataset_dir: str = "./merged_dataset"):
    ds = Path(dataset_dir)
    total = 0
    for split in ["train", "val"]:
        img_dir = ds / "images" / split
        lbl_dir = ds / "labels" / split
        if img_dir.exists():
            n = fix_split(img_dir, lbl_dir)
            print(f"  {split}: {n} files renamed")
            total += n
    print(f"\n✅ Done — {total} files renamed total")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "./merged_dataset"
    print(f"🔧 Fixing non-ASCII filenames in: {path}")
    main(path)
