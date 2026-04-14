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
    "POLE":         0,
    "INSULATOR":    1,
    "CONDUCTOR":    2,
    "CROSSARM":     3,
    "STREET_LIGHT": 4,
    "DTR":          5,
    "OBJECT":       6,
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

            # Generate unique filename
            sample_id  = str(uuid.uuid4())[:12]
            img_file   = f"{sample_id}.jpg"
            label_file = f"{sample_id}.txt"
            img_path   = os.path.join(IMAGES_DIR, img_file)
            lbl_path   = os.path.join(LABELS_DIR, label_file)

            # Save image
            cv2.imwrite(img_path, img)

            # Convert detections → YOLO format
            yolo_lines  = []
            class_counts = {}

            for det in detections:
                label = det.get("label", "OBJECT").upper().replace(" ", "_")
                bbox  = det.get("bbox")

                if not bbox or len(bbox) < 4:
                    continue

                class_id = CLASS_MAP.get(label, CLASS_MAP.get("OBJECT"))
                x1, y1, x2, y2 = bbox

                # Clip to image bounds
                x1 = max(0, min(x1, w)); x2 = max(0, min(x2, w))
                y1 = max(0, min(y1, h)); y2 = max(0, min(y2, h))

                if x2 <= x1 or y2 <= y1:
                    continue

                # YOLO format: class cx cy w h (normalized 0–1)
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h

                yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                class_counts[label] = class_counts.get(label, 0) + 1
                total_class_counts[label] = total_class_counts.get(label, 0) + 1

            if not yolo_lines:
                # Remove image that has no valid annotations
                if os.path.exists(img_path): os.remove(img_path)
                continue

            # Write YOLO label file
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
