"""
training_pipeline.py
====================
Automated Active Learning Pipeline for Infrastructure Detection.

When an asset is approved by an admin:
  1. Images + corrected annotations are exported to YOLO format
  2. Training sample count is tracked in the DB
  3. When threshold is reached, a retrain job is queued

Directory structure created on server:
  training_data/
    images/         ← JPEG images
    labels/         ← YOLO .txt label files
    data.yaml       ← YOLO dataset config
    log.json        ← Training history log
"""

import os
import json
import base64
import uuid
import sqlite3
import threading
from datetime import datetime

# ─── Configuration ────────────────────────────────────────────────────────────
TRAINING_DIR      = "training_data"
IMAGES_DIR        = os.path.join(TRAINING_DIR, "images")
LABELS_DIR        = os.path.join(TRAINING_DIR, "labels")
LOG_PATH          = os.path.join(TRAINING_DIR, "log.json")
YAML_PATH         = os.path.join(TRAINING_DIR, "data.yaml")
RETRAIN_THRESHOLD = 50   # Auto-retrain after this many new approved samples
DB_PATH           = "database.db"

# Map class names → YOLO class index (must match your model's classes.txt)
CLASS_MAP = {
    "POLE":              0,
    "STRUT_POLE":        1,
    "INSULATOR":         2,
    "T_RISING":          3,
    "TAPPING_CHANNEL":   4,
    "SIDE_ARM_CHANNEL":  5,
    "V_CROSS":           6,
    "TOP_CLEAT":         7,
    "CONDUCTOR":         8,
    "STREET_LIGHT":      9,
    "DTR":               10,
    "WIRE_BROKEN":       11,
    "VEGETATION":        12,
    "OBJECT":            13,
}

# ─── Setup ────────────────────────────────────────────────────────────────────
def ensure_dirs():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(LABELS_DIR, exist_ok=True)
    _write_yaml()

def _write_yaml():
    """Write/update YOLO data.yaml with current class list."""
    names = {v: k for k, v in CLASS_MAP.items()}
    lines = [
        f"path: {os.path.abspath(TRAINING_DIR)}",
        f"train: images",
        f"val: images",
        f"nc: {len(CLASS_MAP)}",
        "names:",
    ] + [f"  {i}: {names[i]}" for i in range(len(CLASS_MAP))]
    with open(YAML_PATH, "w") as f:
        f.write("\n".join(lines))

# ─── DB Helpers ───────────────────────────────────────────────────────────────
def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def _init_training_table():
    """Create training_samples table if it doesn't exist."""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS training_samples (
            id          TEXT PRIMARY KEY,
            asset_id    TEXT,
            image_file  TEXT,
            label_file  TEXT,
            class_counts TEXT,
            approved_by TEXT,
            timestamp   TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS training_runs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            triggered_at TEXT,
            sample_count INTEGER,
            status      TEXT DEFAULT 'queued',
            result      TEXT
        )
    """)
    conn.commit()
    conn.close()

# ─── Core Export Function ─────────────────────────────────────────────────────
def export_asset_to_training(asset_id: str, approved_by: str) -> dict:
    """
    Called when admin approves an asset.
    Returns a summary dict with counts.
    """
    ensure_dirs()
    _init_training_table()

    conn = _get_conn()
    image_rows = conn.execute(
        "SELECT * FROM asset_images WHERE asset_id = ?", (asset_id,)
    ).fetchall()
    conn.close()

    if not image_rows:
        return {"exported": 0, "classes": {}}

    total_exported = 0
    total_class_counts = {}
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for row in image_rows:
        try:
            detections = json.loads(row["detections"])
            image_b64  = row["image_b64"]

            if not detections or not image_b64:
                continue

            # Decode image to get real dimensions
            img_bytes = base64.b64decode(image_b64)
            import numpy as np, cv2
            nparr = np.frombuffer(img_bytes, np.uint8)
            img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]

            # 1. Identify all main poles in detections to use as anchor structures
            main_poles = [det for det in detections if "pole" in det.get("label", "").lower() and "strut" not in det.get("label", "").lower() and det.get("confirmed", True)]

            crops = []  # List of tuples: (cx1, cy1, cx2, cy2, crop_suffix)
            if main_poles:
                for idx, p_det in enumerate(main_poles):
                    px1, py1, px2, py2 = p_det["bbox"]
                    pw = px2 - px1
                    ph = py2 - py1
                    
                    assoc_dets = []
                    for det in detections:
                        if not det.get("confirmed", True) or det is p_det:
                            continue
                        bbox = det.get("bbox")
                        if not bbox or len(bbox) < 4:
                            continue
                        bx1, by1, bx2, by2 = bbox
                        
                        # Association checks:
                        # - Component is close horizontally (within 3.5x pole width from center)
                        # - Component lies vertically within the pole top region (from top - 150px to 75% height)
                        p_cx = (px1 + px2) / 2
                        b_cx = (bx1 + bx2) / 2
                        horiz_dist = abs(b_cx - p_cx)
                        is_close_x = horiz_dist < max(150, pw * 3.5)
                        is_in_y = (py1 - 150) <= by2 and by1 <= (py1 + ph * 0.75)
                        
                        if is_close_x and is_in_y:
                            assoc_dets.append(det)
                            
                    if assoc_dets:
                        # --- Stage 1: Structural Crop ---
                        # Filter out conductors/wires and poles from crop anchors to avoid over-expansion
                        hardware_dets = [det for det in assoc_dets if "conductor" not in det.get("label", "").lower() and "cable" not in det.get("label", "").lower() and "pole" not in det.get("label", "").lower()]
                        
                        # Pole top region box: Top 25% of the pole to serve as vertical anchor
                        pole_top_box = [px1, py1, px2, py1 + int(ph * 0.25)]
                        
                        # Merge hardware boxes with pole top anchor
                        anchor_boxes = [det["bbox"] for det in hardware_dets] + [pole_top_box]
                        
                        min_x = min(box[0] for box in anchor_boxes)
                        max_x = max(box[2] for box in anchor_boxes)
                        min_y = min(box[1] for box in anchor_boxes)
                        max_y = max(box[3] for box in anchor_boxes)
                        
                        # Horizontal Expansion using cross-arm spread (minimum of 150px padding)
                        pad_x = int(max(150, (max_x - min_x) * 0.15))
                        cx1 = max(0, min_x - pad_x)
                        cx2 = min(w, max_x + pad_x)
                        
                        # Vertical Expansion around hardware cluster
                        pad_y_top = 150
                        pad_y_bot = 100
                        cy1 = max(0, min_y - pad_y_top)
                        cy2 = min(h, max_y + pad_y_bot)
                        
                        # Aspect Ratio Clamping to prevent ultra-tall vertical crops
                        cw = cx2 - cx1
                        ch = cy2 - cy1
                        if cw > 10 and ch > 1.8 * cw:
                            # Centered clamping to keep hardware cluster in center
                            new_ch = int(1.8 * cw)
                            cy_center = (min_y + max_y) / 2
                            cy1 = max(0, int(cy_center - new_ch / 2))
                            cy2 = min(h, cy1 + new_ch)
                            # Re-align if hit bottom boundary
                            cy1 = max(0, cy2 - new_ch)
                            
                        crops.append((cx1, cy1, cx2, cy2, f"p{idx}"))

                        # --- Stage 2: Hardware-Head Crop (Tight component zoom) ---
                        head_classes = ["insulator", "insulators", "cleat", "clamp", "arm", "cross"]
                        head_dets = [det for det in assoc_dets if any(kw in det.get("label", "").lower() for kw in head_classes)]
                        
                        if head_dets:
                            hx1 = min(det["bbox"][0] for det in head_dets)
                            hy1 = min(det["bbox"][1] for det in head_dets)
                            hx2 = max(det["bbox"][2] for det in head_dets)
                            hy2 = max(det["bbox"][3] for det in head_dets)
                            
                            hw = hx2 - hx1
                            hh = hy2 - hy1
                            
                            # Expand by tight 30% margin
                            pad_hw_x = int(max(80, hw * 0.30))
                            pad_hw_y = int(max(60, hh * 0.30))
                            
                            hcx1 = max(0, hx1 - pad_hw_x)
                            hcx2 = min(w, hx2 + pad_hw_x)
                            hcy1 = max(0, hy1 - pad_hw_y)
                            hcy2 = min(h, hy2 + pad_hw_y)
                            
                            crops.append((hcx1, hcy1, hcx2, hcy2, f"hw{idx}"))
                    else:
                        cx1 = max(0, px1 - int(pw * 2.5))
                        cx2 = min(w, px2 + int(pw * 2.5))
                        cy1 = max(0, py1 - 150)
                        cy2 = min(h, py1 + int(ph * 0.25) + 100)
                        crops.append((cx1, cy1, cx2, cy2, f"p{idx}"))
            else:
                # Fallback to whole image if no main pole detected
                crops.append((0, 0, w, h, "full"))

            for cx1, cy1, cx2, cy2, suffix in crops:
                cw = cx2 - cx1
                ch = cy2 - cy1
                if cw <= 10 or ch <= 10:
                    continue

                crop_img = img[cy1:cy2, cx1:cx2]

                # Generate unique filename for this crop
                sample_id  = f"{str(uuid.uuid4())[:12]}_{suffix}"
                img_file   = f"{sample_id}.jpg"
                label_file = f"{sample_id}.txt"
                img_path   = os.path.join(IMAGES_DIR, img_file)
                lbl_path   = os.path.join(LABELS_DIR, label_file)

                # Save cropped image segment
                cv2.imwrite(img_path, crop_img)

                # Convert remapped detections inside this crop window to standard YOLO segmentation format
                yolo_lines  = []
                class_counts = {}

                for det in detections:
                    if not det.get("confirmed", True):
                        continue

                    label = det.get("label", "OBJECT").upper().replace(" ", "_")
                    bbox  = det.get("bbox")
                    if not bbox or len(bbox) < 4:
                        continue
                        
                    bx1, by1, bx2, by2 = bbox
                    
                    # Compute intersection with the crop window
                    ix1 = max(cx1, bx1)
                    iy1 = max(cy1, by1)
                    ix2 = min(cx2, bx2)
                    iy2 = min(cy2, by2)
                    
                    # If entirely outside the crop window, skip
                    if ix2 <= ix1 or iy2 <= iy1:
                        continue
                        
                    # Proximity check: require at least 15% of detection overlap area or a valid centroid
                    det_area = (bx2 - bx1) * (by2 - by1)
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    if inter_area < 0.15 * det_area:
                        # Exclude highly cut-off or mostly external objects
                        continue

                    class_id = CLASS_MAP.get(label, CLASS_MAP.get("OBJECT"))

                    # Retrieve polygon coordinate list if exists, otherwise construct 4 corners of intersection bbox
                    poly_pts = det.get("polygon")
                    mapped_poly = []
                    
                    if poly_pts and len(poly_pts) >= 3:
                        # Remap original polygon coordinates to cropped window and clip boundaries
                        for pt in poly_pts:
                            if len(pt) >= 2:
                                px, py = pt[0], pt[1]
                                clamped_x = max(cx1, min(cx2, px))
                                clamped_y = max(cy1, min(cy2, py))
                                mapped_poly.append((clamped_x - cx1, clamped_y - cy1))
                                
                        # Remove duplicates
                        cleaned = []
                        for p in mapped_poly:
                            if not cleaned or cleaned[-1] != p:
                                cleaned.append(p)
                        mapped_poly = cleaned
                        
                    # Fallback to bbox corners if polygon is invalid or has too few vertices
                    if len(mapped_poly) < 3:
                        mapped_poly = [
                            (ix1 - cx1, iy1 - cy1),
                            (ix2 - cx1, iy1 - cy1),
                            (ix2 - cx1, iy2 - cy1),
                            (ix1 - cx1, iy2 - cy1)
                        ]

                    # Normalize all mapped coordinates relative to the crop dimensions [0.0, 1.0]
                    norm_coords = []
                    for kx, ky in mapped_poly:
                        norm_coords.append(f"{kx / cw:.6f} {ky / ch:.6f}")
                        
                    # Standard YOLO segmentation format: class_id x1 y1 x2 y2 ...
                    yolo_lines.append(f"{class_id} " + " ".join(norm_coords))
                    class_counts[label] = class_counts.get(label, 0) + 1
                    total_class_counts[label] = total_class_counts.get(label, 0) + 1

                # Handle potential Negative Samples (blank label file for background training)
                if not yolo_lines:
                    import random
                    if random.random() > 0.10:
                        if os.path.exists(img_path):
                            os.remove(img_path)
                        continue
                    else:
                        open(lbl_path, "w").close()
                        print(f"[TrainingPipeline] Exported background sample: {img_file}")
                else:
                    with open(lbl_path, "w") as f:
                        f.write("\n".join(yolo_lines))

                # Log to DB
                conn2 = _get_conn()
                conn2.execute("""
                    INSERT INTO training_samples
                        (id, asset_id, image_file, label_file, class_counts, approved_by, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (sample_id, asset_id, img_file, label_file,
                      json.dumps(class_counts), approved_by, timestamp))
                conn2.commit()
                conn2.close()

                total_exported += 1

        except Exception as e:
            print(f"[TrainingPipeline] Error exporting image: {e}")
            continue

    # Update log file
    _update_log(asset_id, total_exported, total_class_counts, approved_by)

    # Check if we should auto-trigger retrain
    total_new_samples = _count_pending_samples()
    retrain_queued = False
    if total_new_samples >= RETRAIN_THRESHOLD:
        retrain_queued = _queue_retrain(total_new_samples)

    return {
        "exported": total_exported,
        "classes": total_class_counts,
        "total_pool": total_new_samples,
        "retrain_queued": retrain_queued
    }

# ─── Training Stats ───────────────────────────────────────────────────────────
def get_training_stats() -> dict:
    """Returns stats for the Admin dashboard Training Pool panel."""
    _init_training_table()
    conn = _get_conn()

    samples = conn.execute(
        "SELECT class_counts, timestamp FROM training_samples ORDER BY timestamp DESC"
    ).fetchall()

    runs = conn.execute(
        "SELECT * FROM training_runs ORDER BY triggered_at DESC LIMIT 5"
    ).fetchall()

    conn.close()

    # Aggregate class counts
    total_by_class = {}
    for s in samples:
        counts = json.loads(s["class_counts"])
        for k, v in counts.items():
            total_by_class[k] = total_by_class.get(k, 0) + v

    # Check classes below threshold
    weak_classes = [c for c, cnt in total_by_class.items() if cnt < 300]

    return {
        "total_samples":   len(samples),
        "by_class":        total_by_class,
        "weak_classes":    weak_classes,
        "retrain_needed":  len(samples) >= RETRAIN_THRESHOLD,
        "threshold":       RETRAIN_THRESHOLD,
        "recent_runs":     [dict(r) for r in runs],
        "last_approved":   samples[0]["timestamp"] if samples else None,
    }

# ─── Retrain Trigger ──────────────────────────────────────────────────────────
def _count_pending_samples() -> int:
    conn = _get_conn()
    # Count samples added since last successful retrain
    last_run = conn.execute(
        "SELECT triggered_at FROM training_runs WHERE status = 'done' ORDER BY triggered_at DESC LIMIT 1"
    ).fetchone()
    conn.close()

    if last_run:
        conn2 = _get_conn()
        count = conn2.execute(
            "SELECT COUNT(*) as c FROM training_samples WHERE timestamp > ?",
            (last_run["triggered_at"],)
        ).fetchone()["c"]
        conn2.close()
        return count
    else:
        conn2 = _get_conn()
        count = conn2.execute("SELECT COUNT(*) as c FROM training_samples").fetchone()["c"]
        conn2.close()
        return count

def _queue_retrain(sample_count: int) -> bool:
    """Queue a retrain entry in the DB. Actual training started separately."""
    try:
        conn = _get_conn()
        # Avoid duplicate queue entries
        pending = conn.execute(
            "SELECT COUNT(*) as c FROM training_runs WHERE status = 'queued'"
        ).fetchone()["c"]

        if pending == 0:
            conn.execute("""
                INSERT INTO training_runs (triggered_at, sample_count, status)
                VALUES (?, ?, 'queued')
            """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), sample_count))
            conn.commit()
            print(f"[TrainingPipeline] Retrain queued with {sample_count} samples.")
        conn.close()
        return True
    except Exception as e:
        print(f"[TrainingPipeline] Queue error: {e}")
        return False

def _update_log(asset_id, exported, class_counts, approved_by):
    log = []
    if os.path.exists(LOG_PATH):
        try:
            with open(LOG_PATH) as f:
                log = json.load(f)
        except:
            log = []
    log.append({
        "asset_id":    asset_id,
        "exported":    exported,
        "classes":     class_counts,
        "approved_by": approved_by,
        "timestamp":   datetime.now().isoformat()
    })
    with open(LOG_PATH, "w") as f:
        json.dump(log[-500:], f, indent=2)  # Keep last 500 entries

# ─── Initialize on import ─────────────────────────────────────────────────────
ensure_dirs()
_init_training_table()
