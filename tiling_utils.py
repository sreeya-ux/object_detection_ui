import os
import cv2
import json
import numpy as np
from pathlib import Path

def tile_image_and_labels(img_path, lbl_path, output_img_dir, output_lbl_dir, tile_size=1280, overlap=0.1):
    """
    Slices a large image and its YOLO labels into smaller tiles.
    Helpful for training on high-res drone data where objects are small.
    """
    img = cv2.imread(str(img_path))
    if img is None: return
    h, w = img.shape[:2]

    # Load labels
    labels = []
    if os.path.exists(lbl_path):
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    labels.append([int(parts[0])] + [float(x) for x in parts[1:]])

    step = int(tile_size * (1 - overlap))
    
    tile_count = 0
    for y in range(0, h, step):
        for x in range(0, w, step):
            # Define tile bounds (and clip to image size)
            x2, y2 = min(x + tile_size, w), min(y + tile_size, h)
            x1, y1 = max(0, x2 - tile_size), max(0, y2 - tile_size)
            
            tile = img[y1:y2, x1:x2]
            
            # Recalculate labels for this tile
            tile_labels = []
            for cls_id, cx, cy, bw, bh in labels:
                # Convert normalized to absolute pixel coords
                abs_cx, abs_cy = cx * w, cy * h
                abs_bw, abs_bh = bw * w, bh * h
                
                # Check if object center is inside this tile
                if x1 <= abs_cx <= x2 and y1 <= abs_cy <= y2:
                    # New normalized coords relative to tile
                    new_cx = (abs_cx - x1) / tile_size
                    new_cy = (abs_cy - y1) / tile_size
                    new_bw = abs_bw / tile_size
                    new_bh = abs_bh / tile_size
                    tile_labels.append(f"{cls_id} {new_cx:.6f} {new_cy:.6f} {new_bw:.6f} {new_bh:.6f}")

            # Save if tile has objects (or with low prob if blank)
            if tile_labels or np.random.random() < 0.05:
                tile_name = f"{Path(img_path).stem}_t{y1}_{x1}"
                cv2.imwrite(os.path.join(output_img_dir, f"{tile_name}.jpg"), tile)
                with open(os.path.join(output_lbl_dir, f"{tile_name}.txt"), 'w') as f:
                    f.write("\n".join(tile_labels))
                tile_count += 1
                
    return tile_count

if __name__ == "__main__":
    # Test stub
    print("Tiling Utility Ready. Usage: import and call tile_image_and_labels()")
