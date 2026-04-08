"""
count_labels.py
────────────────
Counts both:
  - Images containing each class  (how many files have that class)
  - Total annotations per class   (total bounding boxes)

Usage:
    python count_labels.py
    python count_labels.py ./merged_dataset
"""

import sys
from pathlib import Path
from collections import defaultdict

def count_labels(dataset_dir: str = "./merged_dataset"):
    ds = Path(dataset_dir)

    # Load class names from data.yaml
    yaml_path = ds / "data.yaml"
    class_names = []
    if yaml_path.exists():
        import yaml
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        class_names = data.get("names", [])

    for split in ["train", "val"]:
        lbl_dir = ds / "labels" / split
        if not lbl_dir.exists():
            continue

        label_files    = list(lbl_dir.glob("*.txt"))
        img_count      = defaultdict(int)   # images that contain this class
        annot_count    = defaultdict(int)   # total bounding boxes for this class
        empty_files    = 0
        total_files    = len(label_files)

        for lbl_file in label_files:
            seen_classes = set()
            with open(lbl_file) as f:
                lines = [l.strip() for l in f if l.strip()]

            if not lines:
                empty_files += 1
                continue

            for line in lines:
                parts = line.split()
                if parts:
                    try:
                        cls_id = int(parts[0])
                        annot_count[cls_id] += 1
                        seen_classes.add(cls_id)
                    except ValueError:
                        pass

            for cls_id in seen_classes:
                img_count[cls_id] += 1

        # ── Report ────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  Split: {split.upper()}  ({total_files} label files, {empty_files} empty)")
        print(f"{'='*60}")
        print(f"  {'ID':>3}  {'Class':<20}  {'Images':>8}  {'Annotations':>12}  {'Avg/img':>8}")
        print(f"  {'-'*3}  {'-'*20}  {'-'*8}  {'-'*12}  {'-'*8}")

        all_ids = sorted(set(list(img_count.keys()) + list(annot_count.keys())))

        for cls_id in all_ids:
            name   = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
            imgs   = img_count.get(cls_id, 0)
            annots = annot_count.get(cls_id, 0)
            avg    = annots / imgs if imgs > 0 else 0.0

            # Status icon
            if imgs == 0:
                icon = "❌"
            elif imgs < 500:
                icon = "⚠️ "
            else:
                icon = "✅"

            print(f"  {icon} [{cls_id:>2}] {name:<20}  {imgs:>8,}  {annots:>12,}  {avg:>8.1f}")

        if all_ids:
            min_imgs = min(img_count.get(i, 0) for i in all_ids)
            max_imgs = max(img_count.get(i, 0) for i in all_ids)
            ratio    = max_imgs / min_imgs if min_imgs > 0 else float('inf')
            print(f"\n  Imbalance ratio: {max_imgs:,} / {min_imgs:,} = {ratio:.1f}x")
            if ratio > 5:
                print(f"  ⚠️  Imbalance > 5x — consider more augmentation for rare classes")
            elif ratio > 2:
                print(f"  ℹ️  Moderate imbalance — YOLO handles this well")
            else:
                print(f"  ✅ Well balanced!")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "./merged_dataset"
    print(f"📊 Label Count Report — {path}")
    count_labels(path)
