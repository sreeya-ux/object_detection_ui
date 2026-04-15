"""
retrain_model.py
================
Fine-tune the existing main_best.pt YOLOv8 model on 10K new annotated images.
Supports mixed image channel types: RGB, Grayscale, RGBA, BGR — all normalized
to standard 3-channel RGB before training.

USAGE:
    python retrain_model.py

BEFORE RUNNING:
    1. Place your images in:    raw_dataset/images/   (any channel type)
    2. Place your labels in:    raw_dataset/labels/   (YOLO .txt format)
    3. Run: python retrain_model.py --prepare    ← auto-converts + splits
    4. Then run: python retrain_model.py          ← to start training
"""

import os
import shutil
import cv2
import numpy as np
import random
import argparse
from pathlib import Path

# =============================================================================
# CONFIGURATION — Edit these as needed
# =============================================================================

# Path to your existing model (the one to fine-tune from)
BASE_MODEL = "models/main_best.pt"

# Where your flat 10K images + labels currently live (before splitting)
RAW_IMAGES_DIR = "raw_dataset/images"   # <-- put your 10K images here
RAW_LABELS_DIR = "raw_dataset/labels"   # <-- put your 10K .txt files here

# Final structured dataset directory
DATASET_DIR = "training_data"

# Output: where the new best model will be saved after training
OUTPUT_MODEL_PATH = "models/main_best_v2.pt"

# Training hyperparameters
EPOCHS       = 50       # Increase to 100 for better accuracy (takes longer)
IMAGE_SIZE   = 640      # Standard YOLO input size
BATCH_SIZE   = 16       # Reduce to 8 if you get out-of-memory errors
VAL_SPLIT    = 0.2      # 20% of images used for validation
PATIENCE     = 15       # Early stopping patience (stops if no improvement)
LR0          = 0.001    # Initial learning rate (lower = more careful fine-tuning)
WORKERS      = 4        # Parallel data loader workers (set to 0 on Windows if errors)

# Class names — must match your training labels order
CLASS_NAMES = [
    "POLE_9M",
    "POLE_11M",
    "POLE_8.1M",
    "INS_PIN",
    "INS_DISC",
    "T_RISING",
    "TAPPING_CHANNEL",
    "SIDE_ARM_CHANNEL",
    "V_CROSS",
    "CONDUCTOR",
    "STREET_LIGHT",
    "DTR",
    "WIRE_BROKEN",
    "VEGETATION",
    "OBJECT",
]

# =============================================================================


def prepare_dataset():
    """
    Step 1: Split flat image/label folders into train/val structure.
    Run this once before training if your images are all in one flat folder.
    """
    print("\n[PREPARE] Splitting dataset into train/val...")

    raw_img_path = Path(RAW_IMAGES_DIR)
    raw_lbl_path = Path(RAW_LABELS_DIR)

    if not raw_img_path.exists():
        print(f"\n❌ ERROR: Raw images folder not found: {raw_img_path.absolute()}")
        print("   Create this folder and add your 10K images there.")
        return False

    if not raw_lbl_path.exists():
        print(f"\n❌ ERROR: Raw labels folder not found: {raw_lbl_path.absolute()}")
        print("   Create this folder and add your YOLO .txt label files there.")
        return False

    # Collect all images that have matching labels
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_images = [
        f for f in raw_img_path.iterdir()
        if f.suffix.lower() in image_extensions
    ]

    paired = []
    skipped = 0
    for img_file in all_images:
        label_file = raw_lbl_path / (img_file.stem + ".txt")
        if label_file.exists():
            paired.append((img_file, label_file))
        else:
            skipped += 1

    print(f"   Found {len(paired)} image-label pairs ({skipped} images skipped — no matching label)")

    if len(paired) == 0:
        print("\n❌ No paired image+label files found. Check your folder structure.")
        return False

    # Shuffle and split
    random.seed(42)
    random.shuffle(paired)
    split_idx = int(len(paired) * (1 - VAL_SPLIT))
    train_pairs = paired[:split_idx]
    val_pairs   = paired[split_idx:]

    print(f"   Train: {len(train_pairs)} | Val: {len(val_pairs)}")

    # Create output directories
    for split in ["train", "val"]:
        os.makedirs(os.path.join(DATASET_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(DATASET_DIR, "labels", split), exist_ok=True)

    # Copy files — convert all images to standard 3-channel RGB
    def copy_pairs(pairs, split):
        converted = 0
        for img, lbl in pairs:
            dest_img = os.path.join(DATASET_DIR, "images", split, img.stem + ".jpg")
            dest_lbl = os.path.join(DATASET_DIR, "labels", split, lbl.name)

            # Normalize channel format → 3-channel RGB JPEG
            converted += _normalize_and_save(str(img), dest_img)
            shutil.copy2(lbl, dest_lbl)
        if converted:
            print(f"   ⚡ {converted} images were channel-converted to RGB")

    print("   Copying train files...")
    copy_pairs(train_pairs, "train")
    print("   Copying val files...")
    copy_pairs(val_pairs, "val")

    # Write data.yaml
    _write_yaml()

    print(f"\n✅ Dataset prepared successfully in: {os.path.abspath(DATASET_DIR)}/")
    print(f"   Train images: {len(train_pairs)}")
    print(f"   Val   images: {len(val_pairs)}")
    return True


def _normalize_and_save(src_path: str, dest_path: str) -> int:
    """
    Load any image format (grayscale, RGBA, BGR, RGB) and save as
    a standard 3-channel RGB JPEG for YOLO compatibility.
    Returns 1 if a conversion was needed, 0 if it was already fine.
    """
    img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)  # Load with alpha if present

    if img is None:
        # Fallback: try PIL for exotic formats
        try:
            from PIL import Image as PILImage
            pil_img = PILImage.open(src_path).convert("RGB")
            pil_img.save(dest_path, "JPEG", quality=95)
            return 1
        except Exception as e:
            print(f"   ⚠️  Could not read {src_path}: {e}")
            return 0

    converted = 0
    channels = 1 if img.ndim == 2 else img.shape[2]

    if channels == 1 or img.ndim == 2:
        # Grayscale → RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        converted = 1
    elif channels == 4:
        # RGBA/BGRA → RGB (blend alpha on white background)
        bgr   = img[:, :, :3]
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        white = np.ones_like(bgr, dtype=np.float32) * 255
        img   = (bgr.astype(np.float32) * alpha + white * (1 - alpha)).astype(np.uint8)
        converted = 1
    elif channels == 3:
        # Already 3-channel (OpenCV reads as BGR — YOLO handles BGR fine,
        # but we save as JPEG to ensure uniform format)
        pass  # no conversion needed, just re-save

    cv2.imwrite(dest_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return converted


def _write_yaml():
    """Write the data.yaml config file for YOLO training."""
    yaml_path = os.path.join(DATASET_DIR, "data.yaml")
    dataset_abs = os.path.abspath(DATASET_DIR)

    lines = [
        f"path: {dataset_abs}",
        f"train: images/train",
        f"val:   images/val",
        f"nc: {len(CLASS_NAMES)}",
        "names:",
    ]
    for i, name in enumerate(CLASS_NAMES):
        lines.append(f"  {i}: {name}")

    with open(yaml_path, "w") as f:
        f.write("\n".join(lines))

    print(f"   data.yaml written → {yaml_path}")
    return yaml_path


def check_dataset_ready():
    """Validate that train/val folders exist and have files."""
    train_img = Path(DATASET_DIR) / "images" / "train"
    val_img   = Path(DATASET_DIR) / "images" / "val"
    yaml_file = Path(DATASET_DIR) / "data.yaml"

    ok = True

    if not train_img.exists() or len(list(train_img.iterdir())) == 0:
        print(f"❌ Train images missing: {train_img.absolute()}")
        print("   Run: python retrain_model.py --prepare")
        ok = False

    if not val_img.exists() or len(list(val_img.iterdir())) == 0:
        print(f"❌ Val images missing: {val_img.absolute()}")
        ok = False

    if not yaml_file.exists():
        print(f"❌ data.yaml missing: {yaml_file.absolute()}")
        ok = False

    if not Path(BASE_MODEL).exists():
        print(f"❌ Base model not found: {BASE_MODEL}")
        ok = False

    return ok


def run_training():
    """Fine-tune the existing model on the new dataset."""
    from ultralytics import YOLO

    print("\n" + "="*60)
    print(" FINE-TUNING YOLO MODEL ON 10K ANNOTATED IMAGES")
    print("="*60)
    print(f"  Base model  : {BASE_MODEL}")
    print(f"  Dataset     : {os.path.abspath(DATASET_DIR)}")
    print(f"  Epochs      : {EPOCHS}")
    print(f"  Batch size  : {BATCH_SIZE}")
    print(f"  Image size  : {IMAGE_SIZE}")
    print(f"  Output      : {OUTPUT_MODEL_PATH}")
    print("="*60 + "\n")

    # Load existing model — this is fine-tuning, NOT training from scratch
    model = YOLO(BASE_MODEL)

    yaml_path = os.path.join(DATASET_DIR, "data.yaml")

    # Start training
    results = model.train(
        data        = yaml_path,
        epochs      = EPOCHS,
        imgsz       = IMAGE_SIZE,
        batch       = BATCH_SIZE,
        lr0         = LR0,
        patience    = PATIENCE,
        workers     = WORKERS,
        project     = "training_runs",
        name        = "finetune_10k",
        save        = True,
        save_period = 10,      # Save checkpoint every 10 epochs
        plots       = True,    # Generate training plots
        val         = True,
        device      = "0" if _has_gpu() else "cpu",
        augment     = True,    # Use built-in YOLO augmentations
        # Fine-tuning specific: freeze backbone to preserve learned features
        # Unfreeze after a few epochs to let the whole model adapt
        freeze      = 10,      # Freeze first 10 layers (backbone) initially
        verbose     = True,
    )

    # Copy best model to the target output path
    best_weights = Path("training_runs") / "finetune_10k" / "weights" / "best.pt"
    if best_weights.exists():
        shutil.copy2(best_weights, OUTPUT_MODEL_PATH)
        print(f"\n✅ Training complete!")
        print(f"   New best model saved to: {OUTPUT_MODEL_PATH}")
        print(f"\n📋 To use the new model in your app:")
        print(f"   Copy '{OUTPUT_MODEL_PATH}' to 'models/main_best.pt'")
        print(f"   OR update BASE_MODEL path in your pipeline.py")
    else:
        print(f"\n⚠️  Training finished but best.pt not found at expected path.")
        print(f"   Check: training_runs/finetune_10k/weights/ for saved models.")

    print("\n📊 Training metrics saved in: training_runs/finetune_10k/")
    return results


def _has_gpu():
    """Check if a CUDA GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False


def validate_model(model_path=None):
    """
    Quick validation of the trained model on the val set.
    Run after training to see mAP50 scores by class.
    """
    from ultralytics import YOLO

    target = model_path or OUTPUT_MODEL_PATH
    if not Path(target).exists():
        print(f"❌ Model not found: {target}")
        return

    yaml_path = os.path.join(DATASET_DIR, "data.yaml")
    if not Path(yaml_path).exists():
        print(f"❌ data.yaml not found. Run --prepare first.")
        return

    print(f"\n[VALIDATE] Running validation on: {target}")
    model = YOLO(target)
    metrics = model.val(data=yaml_path, imgsz=IMAGE_SIZE)

    print(f"\n📊 Validation Results:")
    print(f"   mAP50     : {metrics.box.map50:.4f}")
    print(f"   mAP50-95  : {metrics.box.map:.4f}")
    print(f"   Precision : {metrics.box.mp:.4f}")
    print(f"   Recall    : {metrics.box.mr:.4f}")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLO model on 10K images")
    parser.add_argument("--prepare",  action="store_true", help="Prepare dataset (split into train/val)")
    parser.add_argument("--train",    action="store_true", help="Run training (default if no flag)")
    parser.add_argument("--validate", action="store_true", help="Run validation on trained model")
    parser.add_argument("--model",    type=str,            help="Model path for validation")
    args = parser.parse_args()

    if args.prepare:
        prepare_dataset()

    elif args.validate:
        validate_model(args.model)

    else:
        # Default: train
        if not check_dataset_ready():
            print("\n💡 TIP: If your images are in a flat folder, run first:")
            print("   python retrain_model.py --prepare")
        else:
            run_training()
