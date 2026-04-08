"""
augmentation.py
────────────────
Data augmentation for low-count classes.

Two augmentation styles:
  1. Standard     — flips, brightness, noise, blur, shadow, fog
  2. Silhouette   — mimics backlit poles against bright sky
                    (dark pole, desaturated, high contrast edges)

Usage:
  augment_dataset("./merged_dataset", target_count=200)
"""

import cv2
import yaml
import random
import numpy as np
from pathlib import Path
from collections import defaultdict

from files.config import AUG_TARGET_COUNT, AUG_SILHOUETTE_PROB


# ── Augmentation pipelines ────────────────────────────────────

def _standard_augment(img: np.ndarray) -> np.ndarray:
    """
    Standard augmentation: brightness, flip, noise, blur, shadow, fog.
    Does NOT do large rotations — poles must stay vertical.
    """
    out = img.copy()

    # Horizontal flip (poles look same from either side)
    if random.random() < 0.5:
        out = cv2.flip(out, 1)

    # Brightness + contrast
    alpha = random.uniform(0.7, 1.3)   # contrast
    beta  = random.randint(-40, 40)    # brightness
    out   = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    # Small rotation (max ±5° — poles should stay vertical)
    angle = random.uniform(-5, 5)
    h, w  = out.shape[:2]
    M     = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    out   = cv2.warpAffine(out, M, (w, h),
                            borderMode=cv2.BORDER_REFLECT)

    # Gaussian noise
    if random.random() < 0.4:
        noise = np.random.normal(0, random.uniform(5, 20), out.shape)
        out   = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Motion blur (simulates camera shake)
    if random.random() < 0.3:
        k   = random.choice([3, 5])
        ker = np.zeros((k, k))
        ker[k//2, :] = 1.0 / k
        out = cv2.filter2D(out, -1, ker)

    # Random shadow (vertical stripe darkening)
    if random.random() < 0.3:
        h, w  = out.shape[:2]
        x1_s  = random.randint(0, w)
        x2_s  = random.randint(0, w)
        pts   = np.array([[x1_s, 0], [x2_s, 0],
                           [x2_s, h], [x1_s, h]], dtype=np.int32)
        mask  = np.zeros_like(out)
        cv2.fillPoly(mask, [pts], (1, 1, 1))
        out   = np.where(mask == 1,
                          (out * random.uniform(0.4, 0.7)).astype(np.uint8),
                          out)

    # Fog (additive white layer)
    if random.random() < 0.2:
        fog_intensity = random.uniform(0.1, 0.3)
        fog_layer = np.full_like(out, 255)
        out = cv2.addWeighted(out, 1 - fog_intensity,
                               fog_layer, fog_intensity, 0)

    return out


def _silhouette_augment(img: np.ndarray) -> np.ndarray:
    """
    Silhouette augmentation — mimics dawn/dusk backlit pole photos.
    Creates dark poles against bright sky backgrounds.

    This is important because field engineers often photograph poles
    looking upward in bright sunlight, creating silhouette conditions.
    """
    out = img.copy()
    out = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Strongly darken value channel
    out[:, :, 2] *= random.uniform(0.2, 0.45)
    # Desaturate (silhouettes have no colour)
    out[:, :, 1] *= random.uniform(0.0, 0.3)

    out = np.clip(out, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_HSV2BGR)

    # Boost contrast to simulate sky glow
    out = cv2.convertScaleAbs(out, alpha=random.uniform(1.5, 2.5), beta=0)

    # CLAHE to enhance structural edges (pole outline becomes sharper)
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=random.uniform(3.0, 8.0),
        tileGridSize=(8, 8)
    )
    eq = clahe.apply(gray)
    out = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

    # Add slight noise
    noise = np.random.normal(0, 8, out.shape)
    out   = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return out


# ── YOLO label helpers ────────────────────────────────────────

def _rotate_yolo_labels(
    labels: list,
    angle_deg: float,
    img_w: int,
    img_h: int,
) -> list:
    """Rotates YOLO format labels by angle_deg (for small rotations)."""
    out = []
    angle_rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    cx, cy = img_w / 2, img_h / 2

    for label in labels:
        parts = label.strip().split()
        if len(parts) < 5:
            continue
        cls_id = parts[0]
        bx, by, bw, bh = [float(x) for x in parts[1:5]]

        # Convert to pixel corners, rotate, convert back
        px = bx * img_w - cx
        py = by * img_h - cy
        new_px = px * cos_a - py * sin_a
        new_py = px * sin_a + py * cos_a
        new_bx = (new_px + cx) / img_w
        new_by = (new_py + cy) / img_h

        # Clamp to [0, 1]
        new_bx = max(0.0, min(1.0, new_bx))
        new_by = max(0.0, min(1.0, new_by))

        out.append(f"{cls_id} {new_bx:.6f} {new_by:.6f} {bw:.6f} {bh:.6f}\n")
    return out


# ── Main augmentation function ────────────────────────────────

def augment_dataset(
    dataset_dir: str,
    target_count: int = AUG_TARGET_COUNT,
    silhouette_prob: float = AUG_SILHOUETTE_PROB,
    seed: int = 42,
):
    """
    Augments images for classes that have fewer than target_count samples.

    Args:
        dataset_dir    : path to merged YOLO dataset folder
        target_count   : desired minimum images per class
        silhouette_prob: probability of using silhouette style augmentation
        seed           : random seed for reproducibility

    Saves augmented images + labels into the existing train split.
    """
    import math  # imported here to avoid circular dependency
    random.seed(seed)
    np.random.seed(seed)

    ds       = Path(dataset_dir)
    lbl_dir  = ds / "labels" / "train"
    img_dir  = ds / "images" / "train"
    yaml_path = ds / "data.yaml"

    if not yaml_path.exists():
        print(f"ERROR: data.yaml not found at {yaml_path}")
        return

    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    class_names = data.get("names", [])

    # ── Count images per class ────────────────────────────────
    class_img_map = defaultdict(list)  # class_id → [(img_path, lbl_path)]

    for lbl_file in sorted(lbl_dir.glob("*.txt")):
        classes_in_file = set()
        with open(lbl_file) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    try:
                        classes_in_file.add(int(parts[0]))
                    except ValueError:
                        pass

        # Find matching image
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            img_file = img_dir / (lbl_file.stem + ext)
            if img_file.exists():
                for cls_id in classes_in_file:
                    class_img_map[cls_id].append((img_file, lbl_file))
                break

    # ── Report + augment ─────────────────────────────────────
    print("\n📊 Class counts before augmentation:")
    for cls_id, name in enumerate(class_names):
        count = len(class_img_map.get(cls_id, []))
        icon  = "✅" if count >= target_count else "⚠️ " if count >= 20 else "❌"
        print(f"  {icon} [{cls_id:2d}] {name:25s}: {count:4d}")

    total_new = 0

    for cls_id, pairs in class_img_map.items():
        current = len(pairs)
        if current >= target_count:
            continue

        needed = target_count - current
        name   = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        print(f"\n🔄 [{cls_id}] {name}: {current} → target {target_count}")

        generated = 0
        attempts  = 0
        max_tries = needed * 10

        while generated < needed and attempts < max_tries:
            attempts += 1
            img_path, lbl_path = random.choice(pairs)

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Read labels
            with open(lbl_path) as f:
                labels = f.readlines()

            if not labels:
                continue

            # Choose augmentation style
            use_silhouette = random.random() < silhouette_prob

            if use_silhouette:
                aug_img    = _silhouette_augment(img)
                aug_labels = labels  # silhouette doesn't change bbox
                suffix     = f"_aug{total_new:05d}_sil"
            else:
                aug_img    = _standard_augment(img)
                aug_labels = labels  # small rotation labels approximated
                suffix     = f"_aug{total_new:05d}"

            # Save
            new_img = img_dir / (img_path.stem + suffix + img_path.suffix)
            new_lbl = lbl_dir / (img_path.stem + suffix + ".txt")

            cv2.imwrite(str(new_img), aug_img)
            with open(new_lbl, "w") as f:
                f.writelines(aug_labels)

            generated  += 1
            total_new  += 1

        print(f"   Generated {generated} new images")

    print(f"\n✅ Augmentation complete — {total_new} new images total")

    # ── Re-report final counts ────────────────────────────────
    print("\n📊 Class counts after augmentation:")
    class_img_map2 = defaultdict(int)
    for lbl_file in lbl_dir.glob("*.txt"):
        with open(lbl_file) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    try:
                        class_img_map2[int(parts[0])] += 1
                    except ValueError:
                        pass
    for cls_id, name in enumerate(class_names):
        count = class_img_map2.get(cls_id, 0)
        icon  = "✅" if count >= target_count else "⚠️ "
        print(f"  {icon} [{cls_id:2d}] {name:25s}: {count:4d}")


if __name__ == "__main__":
    import sys
    from files.config import AUG_TARGET_COUNT, AUG_SILHOUETTE_PROB
    
    # Path to the merged dataset
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "./merged_dataset"
    
    print(f"🚀 DATA AUGMENTATION START")
    print(f"📂 Dataset: {dataset_path}")
    print(f"🎯 Target:  {AUG_TARGET_COUNT} per class")
    
    augment_dataset(
        dataset_dir     = dataset_path,
        target_count    = AUG_TARGET_COUNT,
        silhouette_prob = AUG_SILHOUETTE_PROB
    )
