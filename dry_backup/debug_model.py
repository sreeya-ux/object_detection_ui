import sys
from ultralytics import YOLO
import torch

# Usage: python debug_model.py [model_path] [img_path] [conf] [imgsz]
model_path = sys.argv[1] if len(sys.argv) > 1 else r"D:\NEW_ASAKTA\dry\best_whole.pt"
img_path   = sys.argv[2] if len(sys.argv) > 2 else "test.jpeg"
conf_val   = float(sys.argv[3]) if len(sys.argv) > 3 else 0.10
img_size   = int(sys.argv[4]) if len(sys.argv) > 4 else 640

print(f"Loading {model_path}...")
model = YOLO(model_path)

print(f"\nRunning inference on {img_path} with conf={conf_val} imgsz={img_size}...")
results = model(img_path, conf=conf_val, imgsz=img_size)

print("\nDetections:")
for r in results:
    if r.boxes:
        for b in r.boxes:
            cls_id = int(b.cls[0])
            cls_name = model.names[cls_id]
            c = float(b.conf[0])
            print(f"  {cls_name} (ID: {cls_id}) - Conf: {c:.3f}")
    else:
        print("  No boxes detected.")
