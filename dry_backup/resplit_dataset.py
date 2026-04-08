"""
resplit_dataset.py
───────────────────
Takes ALL images in a merged YOLO dataset (train + val combined)
and creates a fresh stratified 80/20 split.

This fixes skewed val splits where rare classes (crossarm, street_light)
end up with almost no validation examples.

Usage:
    python resplit_dataset.py
    python resplit_dataset.py ./merged_dataset 0.2
"""

import sys
import shutil
import random
from pathlib import Path
from collections import defaultdict


def get_dominant_class(lbl_path: Path) -> int:
    """Returns the most frequent class ID in a label file."""
    counts = defaultdict(int)
    try:
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    counts[int(parts[0])] += 1
    except Exception:
        return -1
    if not counts:
        return -1
    return max(counts, key=counts.get)


def resplit(dataset_dir: str = "./merged_dataset", val_ratio: float = 0.2, seed: int = 42):
    random.seed(seed)
    ds = Path(dataset_dir)

    img_train = ds / "images" / "train"
    img_val   = ds / "images" / "val"
    lbl_train = ds / "labels" / "train"
    lbl_val   = ds / "labels" / "val"

    # ── Collect ALL image-label pairs from both splits ────────
    print("📦 Collecting all image-label pairs...")
    all_pairs = []  # (img_path, lbl_path, dominant_class)

    for img_dir, lbl_dir in [(img_train, lbl_train), (img_val, lbl_val)]:
        if not img_dir.exists():
            continue
        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            dom_cls  = get_dominant_class(lbl_path) if lbl_path.exists() else -1
            all_pairs.append((img_path, lbl_path if lbl_path.exists() else None, dom_cls))

    print(f"  Total pairs: {len(all_pairs)}")

    # ── Stratified split by dominant class ───────────────────
    by_class = defaultdict(list)
    for pair in all_pairs:
        by_class[pair[2]].append(pair)

    new_train, new_val = [], []
    for cls_id, pairs in by_class.items():
        random.shuffle(pairs)
        n_val = max(1, int(len(pairs) * val_ratio))
        new_val.extend(pairs[:n_val])
        new_train.extend(pairs[n_val:])

    print(f"  New train: {len(new_train)}  |  New val: {len(new_val)}")

    # ── Move everything to a temp staging area ────────────────
    print("\n🔄 Rebuilding splits...")
    tmp = ds / "_tmp_resplit"
    for split_name, pairs in [("train", new_train), ("val", new_val)]:
        (tmp / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (tmp / "labels" / split_name).mkdir(parents=True, exist_ok=True)
        for img_path, lbl_path, _ in pairs:
            dst_img = tmp / "images" / split_name / img_path.name
            shutil.move(str(img_path), str(dst_img))   # move = fast (no byte copy)
            if lbl_path and lbl_path.exists():
                dst_lbl = tmp / "labels" / split_name / (img_path.stem + ".txt")
                shutil.move(str(lbl_path), str(dst_lbl))

    # ── Replace the original splits ───────────────────────────
    for split_name in ["train", "val"]:
        shutil.rmtree(ds / "images" / split_name, ignore_errors=True)
        shutil.rmtree(ds / "labels" / split_name, ignore_errors=True)
        shutil.move(str(tmp / "images" / split_name), str(ds / "images" / split_name))
        shutil.move(str(tmp / "labels" / split_name), str(ds / "labels" / split_name))

    shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n✅ Resplit complete!")
    print(f"   Train: {len(new_train)} images")
    print(f"   Val:   {len(new_val)} images ({val_ratio*100:.0f}%)")
    print(f"\n   Run: python count_labels.py   to verify the new distribution")


if __name__ == "__main__":
    path      = sys.argv[1] if len(sys.argv) > 1 else "./merged_dataset"
    val_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
    print(f"🔀 Resplitting: {path}  (val={val_ratio*100:.0f}%)")
    resplit(path, val_ratio)
