"""
train.py
─────────
Trains the YOLOv11 component detection model.

Two training modes:
  --mode standard   → regular YOLOv11 detection (bbox only)
  --mode obb        → YOLOv11-OBB (bbox + angle, more accurate)

OBB gives you the angle of each object — used by insulator
and pole orientation classifiers. Requires OBB-format labels.

Usage:
  python train.py --data ./merged_dataset/data.yaml
  python train.py --data ./merged_dataset/data.yaml --mode obb --epochs 80
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def train(
    data_yaml: str,
    mode: str        = "standard",
    epochs: int      = 100,
    imgsz: int       = 640,
    batch: int       = 16,
    patience: int    = 30,
    device: str      = "auto",
    name: str        = "component_v1",
    resume: bool     = False,
    resume_path: str = None,
):
    """
    Trains YOLOv11 on the 5-class component dataset.

    Args:
        data_yaml   : path to data.yaml
        mode        : "standard" (bbox only) — OBB needs re-labeled data
        epochs      : training epochs (100 recommended for real convergence)
        imgsz       : image size (640 standard, 1280 if GPU allows)
        batch       : batch size (16 for 8GB VRAM, 8 for 4GB, -1 for auto)
        patience    : early stop patience (30 gives model time to recover)
        device      : "auto" detects GPU, "cpu" forces CPU, "0" for GPU 0
        name        : run name (saved under runs/detect/<name>/)
        resume      : resume from a previous checkpoint
        resume_path : path to .pt checkpoint to resume from
    """
    import torch

    # ── Auto GPU/CPU detection ────────────────────────────────
    if device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"
        print(f"Device auto-detected: {'GPU (cuda:0)' if device == '0' else 'CPU (slow)'}")

    # ── Model selection ───────────────────────────────────────
    if resume and resume_path:
        print(f"Resuming from: {resume_path}")
        model = YOLO(resume_path)
    elif mode == "obb":
        print("Loading YOLOv11s-OBB (pretrained on DOTA)...")
        model = YOLO("yolo11s-obb.pt")
    else:
        print("Loading YOLOv11s standard (pretrained on COCO)...")
        model = YOLO("yolo11s.pt")

    # ── Training config ───────────────────────────────────────
    train_args = dict(
        data     = data_yaml,
        epochs   = epochs,
        imgsz    = imgsz,
        batch    = batch,
        patience = patience,
        device   = device,
        name     = name,
        exist_ok = True,

        # ── Speed optimizations (zero quality loss) ───────────
        # AMP = Automatic Mixed Precision (FP16 on GPU)
        # ~40% faster, identical accuracy, industry standard
        amp     = True,

        # Cache images in RAM — eliminates disk I/O bottleneck
        # Needs ~3-5GB RAM for 9k images. Use "disk" if low RAM.
        cache   = "ram",

        # Parallel data loading workers
        workers = 8,

        # Rectangular batches — no wasted padding pixels
        # Slight speed gain, negligible quality difference
        rect    = False,   # keep False — mosaic needs square batches

        # ── Learning rate schedule ────────────────────────────
        cos_lr         = True,
        warmup_epochs  = 3,
        warmup_bias_lr = 0.1,

        # ── Flip augmentation ─────────────────────────────────
        flipud = 0.0,  # NO vertical flip — poles must stay vertical
        fliplr = 0.5,  # horizontal flip OK (poles look same from either side)

        # ── Rotation ─────────────────────────────────────────
        # Small rotation — simulates slightly tilted camera
        # 0 for OBB (rotation would invalidate angle labels)
        degrees = 0.0 if mode == "obb" else 5.0,

        # ── Colour / exposure augmentation ───────────────────
        # These together simulate:
        #   - Dawn/dusk lighting (hsv_v low end)
        #   - Overcast vs. bright sun (hsv_s variation)
        #   - Silhouette conditions (dark pole, bright sky)
        hsv_h    = 0.015,  # hue shift ±1.5% (slight colour cast)
        hsv_s    = 0.7,    # saturation ±70% (overcast → vivid)
        hsv_v    = 0.5,    # brightness ±50%  ← silhouette range increase

        # ── CLAHE — contrast enhancement ─────────────────────
        # Mimics backlit pole silhouette conditions (dark pole, bright sky)
        # Model learns to detect structure even when colours wash out
        clahe    = True,

        # ── Blur ─────────────────────────────────────────────
        # Simulates camera shake from field engineers shooting handheld
        blur     = 0.01,    # Gaussian blur applied randomly

        # ── Scale variation ───────────────────────────────────
        # Poles appear at very different sizes (close-up vs. far away)
        scale    = 0.5,

        # ── Mosaic ───────────────────────────────────────────
        # Combines 4 images — excellent for small object detection
        # (insulators and crossarms are small relative to the full image)
        mosaic   = 1.0 if mode != "obb" else 0.5,

        # ── Random erasing ────────────────────────────────────
        # Simulates partial occlusion: tree branches, other poles, wires
        erasing  = 0.4,

        # ── Copy-paste ────────────────────────────────────────
        # Pastes extra objects into scenes — helps rare classes
        # (crossarm, street_light) appear in more varied backgrounds
        copy_paste = 0.1,

        plots    = True,   # save training curves
        save     = True,   # save best + last checkpoints
        verbose  = True,
    )

    print(f"\nTraining config:")
    print(f"  Mode      : {mode}")
    print(f"  Epochs    : {epochs}  (patience={patience})")
    print(f"  Batch     : {batch}")
    print(f"  ImgSize   : {imgsz}")
    print(f"  Device    : {device}")
    print(f"  Data      : {data_yaml}")
    print(f"  Run name  : {name}")
    print(f"  Output    : runs/detect/{name}/")

    results = model.train(**train_args)

    best = Path(f"runs/detect/{name}/weights/best.pt")
    if not best.exists():
        best = Path(f"runs/obb/{name}/weights/best.pt")

    print(f"\n✅ Training complete!")
    print(f"   Best model : {best}")
    if hasattr(results, "results_dict"):
        mAP = results.results_dict.get("metrics/mAP50(B)", "see plots")
        print(f"   mAP@50     : {mAP}")

    return results, str(best)


def evaluate(model_path: str, data_yaml: str, device: str = "0"):
    """Runs validation and prints per-class results."""
    from files.config import COMPONENT_CLASSES
    model    = YOLO(model_path)
    val_res  = model.val(data=data_yaml, device=device)

    print(f"\n📊 Overall mAP@50:    {val_res.box.map50:.3f}")
    print(f"   Overall mAP@50-95: {val_res.box.map:.3f}")
    print(f"   Precision:         {val_res.box.mp:.3f}")
    print(f"   Recall:            {val_res.box.mr:.3f}")

    print("\n📋 Per-class mAP@50:")
    if hasattr(val_res.box, "ap_class_index"):
        for i, idx in enumerate(val_res.box.ap_class_index):
            ap   = val_res.box.ap50[i]
            # Use COMPONENT_CLASSES for component detection model
            name = COMPONENT_CLASSES[idx] if idx < len(COMPONENT_CLASSES) else str(idx)
            bar  = "█" * int(ap * 20) + "░" * (20 - int(ap * 20))
            icon = "✅" if ap > 0.5 else "⚠️ " if ap > 0.25 else "❌"
            print(f"  {icon} [{idx:2d}] {name:22s} {ap:.3f} |{bar}|")


def export(model_path: str, formats: list = None):
    """Exports model to ONNX and TFLite for deployment."""
    if formats is None:
        formats = ["onnx"]

    model = YOLO(model_path)
    for fmt in formats:
        print(f"Exporting to {fmt}...")
        model.export(format=fmt)
        print(f"  ✅ {fmt} export done")


# ── CLI ───────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 33kV infrastructure model")
    parser.add_argument("--data",     required=True, help="Path to data.yaml")
    parser.add_argument("--mode",     default="standard", choices=["standard", "obb"])
    parser.add_argument("--epochs",   type=int, default=50)
    parser.add_argument("--batch",    type=int, default=16)
    parser.add_argument("--imgsz",    type=int, default=640)
    parser.add_argument("--device",   default="0")
    parser.add_argument("--name",     default="infra_v1")
    parser.add_argument("--resume",   action="store_true")
    parser.add_argument("--resume-path", default=None)
    parser.add_argument("--eval",     action="store_true", help="Evaluate after training")
    parser.add_argument("--export",   action="store_true", help="Export to ONNX after training")
    args = parser.parse_args()

    results, best_path = train(
        data_yaml   = args.data,
        mode        = args.mode,
        epochs      = args.epochs,
        batch       = args.batch,
        imgsz       = args.imgsz,
        device      = args.device,
        name        = args.name,
        resume      = args.resume,
        resume_path = args.resume_path,
    )

    if args.eval:
        evaluate(best_path, args.data, args.device)

    if args.export:
        export(best_path)
