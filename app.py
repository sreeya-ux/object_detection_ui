from flask import Flask, render_template, request, jsonify, session, redirect, url_for, make_response, send_file, send_from_directory
from werkzeug.security import check_password_hash, generate_password_hash
import csv
import io
import base64
import uuid
import time
import re
import pathlib
import logging
import math
from datetime import datetime
import sqlite3
import json
import os
import ast
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict

from config import DB_TYPE, DB_NAME, PG_HOST, PG_PORT, PG_USER, PG_PASS, PG_DB
from worker import enqueue_video_job, get_video_job_status, start_video_workers, update_video_job

from flask_cors import CORS
import cv2

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except Exception as exc:
    psycopg2 = None
    RealDictCursor = None
    print(f"[startup] PostgreSQL driver unavailable, SQLite fallback enabled: {exc}")

logger = logging.getLogger(__name__)
DEBUG_DIR = pathlib.Path("debug_crops")
DEBUG_DIR.mkdir(exist_ok=True)


def _save_debug_crop(img, tag=""):
    fname = DEBUG_DIR / f"{tag}_{int(time.time()*1000)}.jpg"
    cv2.imwrite(str(fname), img)
    logger.info(f"[OCR] saved debug crop -> {fname}")

# =========================
# GLOBAL INITIALIZATION
# =========================
app = Flask(__name__)
CORS(app) # Enable CORS for all routes
app.secret_key = "secret_key_for_session" # In production, use a strong random key
DB_PATH = 'database.db'
UPLOADS_FOLDER = 'uploads'
if not os.path.exists(UPLOADS_FOLDER):
    os.makedirs(UPLOADS_FOLDER)

@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(UPLOADS_FOLDER, filename)

pipeline_engine = None
unet_model = None
device = None
rapidocr_engine = None
video_component_model = None
MODEL_PATHS = {
    "pole": "models/best (2).pt",
    "components": "models/channel_12class_v2.pt",
    "insulator": "models/insulator_model.pt",
    "shed": "models/shed_model.pt",
    "conductor_unet": "models/conductor_unet.pth",
    "video_component": "models/component_model.pt",
}

VIDEO_RESULTS_FOLDER = os.path.join("static", "results")
os.makedirs(VIDEO_RESULTS_FOLDER, exist_ok=True)
VIDEO_LOG_FILE = "video_processing.log"
VIDEO_PROGRESS = {}
VIDEO_INFER_IMGSZ = int(os.environ.get("VIDEO_INFER_IMGSZ", "416") or 416)
VIDEO_OUTPUT_MAX_WIDTH = int(os.environ.get("VIDEO_OUTPUT_MAX_WIDTH", "960") or 960)
VIDEO_SAMPLE_FPS = max(0.1, float(os.environ.get("VIDEO_SAMPLE_FPS", "1") or 1))
MIN_POLE_VISIBLE_SECONDS = 2.0
TEMPORAL_GAP_MAX_SECONDS = 3.0
IMAGE_DEFAULT_DISPLAY_THRESHOLD = 0.40
IMAGE_SMALL_HARDWARE_DISPLAY_THRESHOLD = 0.25
IMAGE_SMALL_HARDWARE_CLASSES = {"street_light", "special_clamp"}
DATASET_INVENTORY_ROOTS = [
    "training_data",
    "training_data_component",
    "training_data_hardware",
    "dataset_channels",
    "dataset_combined",
]
DATASET_STATS_ROOTS = [
    "training_data_component",
    "dataset_combined",
]
CHANNEL_DATASET_CLASS_MAP = {
    0: "INSULATORS",
    1: "V_CROSS_ARM",
    2: "TAPPING_ARM",
    3: "TOP_CLEAT",
    4: "SIDE_ARM",
    5: "T_RISING",
    6: "SPECIAL_CLAMP",
    7: "STREET_LIGHT",
    8: "STAY_SET",
    9: "BOX_ARM",
    10: "AB_SWITCH",
    11: "DTR",
}
COMPONENT_DATASET_CLASS_MAP = {
    0: "MAIN_POLE",
    1: "STRUT_POLE",
}
HARDWARE_DATASET_CLASS_MAP = {
    0: "INSULATOR",
    1: "CROSSARM",
    2: "CONDUCTOR",
}

def log_video(message):
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    try:
        with open(VIDEO_LOG_FILE, "a", encoding="utf-8") as log_file:
            log_file.write(line + "\n")
    except Exception as exc:
        print(f"[VIDEO] Unable to write video log file: {exc}", flush=True)

def set_video_progress(job_id, percent, message, status="processing"):
    if not job_id:
        return
    progress = max(0, min(100, int(percent)))
    VIDEO_PROGRESS[job_id] = {
        "progress": progress,
        "percent": max(0, min(100, int(percent))),
        "message": message,
        "status": status,
        "updated_at": time.time(),
    }
    updates = {"progress": progress, "message": message}
    if status in {"error", "failed"}:
        updates["status"] = "failed"
    elif status in {"queued", "processing"}:
        updates["status"] = status
    update_video_job(job_id, **updates)

def validate_model_paths(*keys):
    missing = []
    for key in keys:
        path = MODEL_PATHS[key]
        if not os.path.exists(path):
            missing.append(f"{key}: {path}")
    if missing:
        raise FileNotFoundError(
            "Missing local model file(s): "
            + ", ".join(missing)
            + ". Check the models/ folder; network downloads are disabled for this app."
        )

def image_display_threshold(label):
    normalized = str(label or "").strip().lower()
    return (
        IMAGE_SMALL_HARDWARE_DISPLAY_THRESHOLD
        if normalized in IMAGE_SMALL_HARDWARE_CLASSES
        else IMAGE_DEFAULT_DISPLAY_THRESHOLD
    )

def _count_dataset_files(folder):
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    label_exts = {".txt", ".json"}
    image_count = 0
    annotation_count = 0
    if not os.path.isdir(folder):
        return image_count, annotation_count
    for root, _, files in os.walk(folder):
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext in image_exts:
                image_count += 1
            elif ext in label_exts:
                annotation_count += 1
    return image_count, annotation_count

def _count_unique_dataset_images(folders):
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    unique_hashes = set()
    fallback_paths = set()
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        for root, _, files in os.walk(folder):
            for filename in files:
                if os.path.splitext(filename)[1].lower() not in image_exts:
                    continue
                image_path = os.path.join(root, filename)
                try:
                    import hashlib
                    with open(image_path, "rb") as handle:
                        unique_hashes.add(hashlib.md5(handle.read()).hexdigest())
                except Exception:
                    fallback_paths.add(os.path.abspath(image_path))
    return len(unique_hashes) + len(fallback_paths)

def _normalise_stats_label(label):
    return str(label or "OBJECT").strip().upper().replace(" ", "_")

def _parse_dataset_names_yaml(yaml_path):
    if not yaml_path or not os.path.exists(yaml_path):
        return {}
    try:
        lines = pathlib.Path(yaml_path).read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception as exc:
        print(f"[TRAINING-STATS] Could not read class map {yaml_path}: {exc}", flush=True)
        return {}

    names = {}
    for idx, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        if not stripped.startswith("names"):
            continue

        inline = stripped.split(":", 1)[1].strip() if ":" in stripped else ""
        if inline:
            try:
                parsed = ast.literal_eval(inline)
                if isinstance(parsed, list):
                    return {i: _normalise_stats_label(name) for i, name in enumerate(parsed)}
                if isinstance(parsed, dict):
                    return {int(k): _normalise_stats_label(v) for k, v in parsed.items()}
            except Exception:
                pass

        for child in lines[idx + 1:]:
            if not child.startswith((" ", "\t")):
                break
            child = child.strip()
            if ":" not in child:
                continue
            key, value = child.split(":", 1)
            try:
                names[int(key.strip())] = _normalise_stats_label(value.strip().strip("'\""))
            except ValueError:
                continue
        break
    return names

def _dataset_class_map_for(folder):
    yaml_map = _parse_dataset_names_yaml(os.path.join(folder, "data.yaml"))
    if yaml_map:
        return yaml_map
    if folder in {"dataset_channels", "dataset_combined"}:
        channel_yaml_map = _parse_dataset_names_yaml("channels.yaml")
        return channel_yaml_map or CHANNEL_DATASET_CLASS_MAP
    if folder == "training_data_component":
        return COMPONENT_DATASET_CLASS_MAP
    if folder == "training_data_hardware":
        return HARDWARE_DATASET_CLASS_MAP
    return {}

def _scan_dataset_label_stats(folder):
    class_map = _dataset_class_map_for(folder)
    by_class = Counter()
    images_per_class = Counter()
    unknown_ids = Counter()
    label_files = 0

    if not os.path.isdir(folder):
        return by_class, images_per_class, unknown_ids, label_files, class_map

    for root, _, files in os.walk(folder):
        for filename in files:
            if os.path.splitext(filename)[1].lower() != ".txt":
                continue
            label_files += 1
            labels_in_file = set()
            label_path = os.path.join(root, filename)
            try:
                with open(label_path, "r", encoding="utf-8", errors="ignore") as handle:
                    for raw_line in handle:
                        parts = raw_line.strip().split()
                        if not parts:
                            continue
                        try:
                            class_id = int(float(parts[0]))
                        except ValueError:
                            continue
                        label = class_map.get(class_id)
                        if not label:
                            unknown_ids[class_id] += 1
                            continue
                        by_class[label] += 1
                        labels_in_file.add(label)
            except Exception as exc:
                print(f"[TRAINING-STATS] Could not scan label file {label_path}: {exc}", flush=True)
                continue
            for label in labels_in_file:
                images_per_class[label] += 1
    return by_class, images_per_class, unknown_ids, label_files, class_map

def build_training_dashboard_stats(stats=None):
    stats = dict(stats or {})
    dataset_by_class = Counter()
    dataset_images_per_class = Counter()
    unknown_by_dataset = {}
    scanned_label_files = 0

    datasets = []
    for folder in DATASET_INVENTORY_ROOTS:
        if os.path.isdir(folder):
            images, annotations = _count_dataset_files(folder)
            datasets.append({
                "name": folder,
                "path": os.path.abspath(folder),
                "count": images,
                "images": images,
                "annotations": annotations,
                "included_in_stats": folder in DATASET_STATS_ROOTS,
            })

    for folder in DATASET_STATS_ROOTS:
        by_class, images_per_class, unknown_ids, label_files, class_map = _scan_dataset_label_stats(folder)
        dataset_by_class.update(by_class)
        dataset_images_per_class.update(images_per_class)
        scanned_label_files += label_files
        if unknown_ids:
            unknown_by_dataset[folder] = {str(k): v for k, v in sorted(unknown_ids.items())}
            print(
                f"[TRAINING-STATS] {folder} has label ids missing from its class map: "
                f"{dict(sorted(unknown_ids.items()))}",
                flush=True
            )

    by_class = dict(sorted(dataset_by_class.items(), key=lambda item: item[1], reverse=True))
    stats["total_samples"] = _count_unique_dataset_images(DATASET_STATS_ROOTS)
    stats["by_class"] = by_class
    stats["total_classes"] = len(by_class)
    stats["total_annotations"] = int(sum(by_class.values()))
    stats["avg_confidence"] = 0.85
    stats["class_confidence"] = {label: 0.85 for label in by_class}
    stats["images_per_class"] = dict(dataset_images_per_class)
    stats["datasets"] = datasets
    stats["stats_source"] = "dataset_folders"
    stats["scanned_label_files"] = scanned_label_files
    stats["unknown_label_ids"] = unknown_by_dataset

    model_descriptions = {
        "pole": "Pole and strut pole detector",
        "components": "Hardware and component detector",
        "insulator": "Insulator detector",
        "shed": "Shed counter",
        "conductor_unet": "Conductor segmentation model",
        "video_component": "Video pole detector",
    }
    models = []
    for key, path in MODEL_PATHS.items():
        exists = os.path.exists(path)
        models.append({
            "key": key,
            "label": os.path.basename(path),
            "path": path,
            "description": model_descriptions.get(key, key.replace("_", " ").title()),
            "exists": exists,
            "size_mb": round(os.path.getsize(path) / (1024 * 1024), 1) if exists else None,
            "num_classes": 0,
            "classes": [],
        })
    stats["models"] = models
    stats["status"] = "ok"
    print(
        "[TRAINING-STATS] "
        f"source=dataset_folders samples={stats.get('total_samples', 0)} "
        f"classes={stats['total_classes']} annotations={stats['total_annotations']} "
        f"label_files={scanned_label_files} datasets={len(stats['datasets'])} models={len(stats['models'])}",
        flush=True
    )
    return stats

def safe_remove_file(path, label="temp file"):
    """Best-effort cleanup for Windows, where model readers may release files late."""
    if not path or not os.path.exists(path):
        return
    for attempt in range(3):
        try:
            os.remove(path)
            return
        except PermissionError as exc:
            if attempt < 2:
                time.sleep(0.2)
                continue
            print(f"[cleanup] Could not remove locked {label}: {path} ({exc})", flush=True)
            return
        except OSError as exc:
            print(f"[cleanup] Could not remove {label}: {path} ({exc})", flush=True)
            return

def load_detection_models(load_unet=False):
    """Load heavy AI dependencies only when image prediction is requested."""
    global pipeline_engine, unet_model, device
    if pipeline_engine is not None and (not load_unet or unet_model is not None):
        return

    import torch
    from pipeline import InfrastructurePipeline

    required = ["pole", "components", "shed", "insulator"]
    if load_unet:
        required.append("conductor_unet")
    validate_model_paths(*required)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if pipeline_engine is None:
        pipeline_engine = InfrastructurePipeline(
            comp_model=MODEL_PATHS["pole"],
            hardware_model=MODEL_PATHS["components"],
            shed_model=MODEL_PATHS["shed"],
            insulator_model=MODEL_PATHS["insulator"]
        )

    if load_unet and unet_model is None:
        import segmentation_models_pytorch as smp
        unet_model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
        unet_model.load_state_dict(torch.load(MODEL_PATHS["conductor_unet"], map_location="cpu"))
        unet_model.eval()
        unet_model.to(device)

def load_video_component_model():
    """Load only the component model used for video pole/strut-pole detection."""
    global video_component_model
    if video_component_model is not None:
        return video_component_model

    validate_model_paths("video_component")
    log_video(f"[VIDEO] Loading video model: {MODEL_PATHS['video_component']}")
    from pipeline import _safe_yolo_load
    video_component_model = _safe_yolo_load(MODEL_PATHS["video_component"])
    log_video("[VIDEO] component_model.pt loaded for video detection")
    return video_component_model

def load_training_pipeline():
    from training_pipeline import export_asset_to_training, get_training_stats
    return export_asset_to_training, get_training_stats

def load_report_generators():
    from report_generator import generate_asset_pdf, generate_asset_excel, generate_global_excel, generate_global_pdf
    return generate_asset_pdf, generate_asset_excel, generate_global_excel, generate_global_pdf

# =========================
# DATABASE HELPERS
# =========================
class DBConn:
    """Wrapper to make PostgreSQL behave like SQLite (execute directly on conn)."""
    def __init__(self, conn, is_pg=False):
        self.conn = conn
        self.is_pg = is_pg
    def execute(self, sql, params=()):
        if self.is_pg:
            # PostgreSQL uses %s instead of ?
            sql = sql.replace("?", "%s")
            cur = self.conn.cursor(cursor_factory=RealDictCursor)
        else:
            cur = self.conn.cursor()
        cur.execute(sql, params)
        return cur
    def commit(self): self.conn.commit()
    def rollback(self): self.conn.rollback()
    def close(self): self.conn.close()

def ensure_sqlite_schema(conn):
    """Create or repair the local SQLite fallback schema without deleting data."""
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS assets (
            id TEXT PRIMARY KEY,
            worker_name TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            timestamp TEXT NOT NULL,
            asset_class TEXT,
            voltage TEXT,
            reason TEXT,
            pole_id TEXT
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS asset_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_id TEXT NOT NULL,
            image_b64 TEXT NOT NULL,
            detections TEXT NOT NULL,
            pole_angle FLOAT DEFAULT 0.0,
            FOREIGN KEY (asset_id) REFERENCES assets(id) ON DELETE CASCADE
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS activity_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_name TEXT NOT NULL,
            action TEXT NOT NULL,
            details TEXT,
            timestamp TEXT NOT NULL
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS training_samples (
            id TEXT PRIMARY KEY,
            asset_id TEXT NOT NULL,
            image_file TEXT,
            label_file TEXT,
            class_counts TEXT,
            approved_by TEXT,
            timestamp TEXT NOT NULL
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS training_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            triggered_at TEXT,
            sample_count INTEGER,
            status TEXT DEFAULT 'queued',
            result TEXT
        )
    ''')

    asset_cols = [row["name"] for row in conn.execute("PRAGMA table_info(assets)").fetchall()]
    if "pole_id" not in asset_cols:
        conn.execute("ALTER TABLE assets ADD COLUMN pole_id TEXT")

    default_users = [
        ("admin 1", generate_password_hash("admin@asakta"), "admin"),
        ("user 1", generate_password_hash("1233@asakta"), "user"),
        ("admin", generate_password_hash("admin@asakta"), "admin"),
    ]
    conn.executemany(
        "INSERT OR IGNORE INTO users (username, password, role) VALUES (?, ?, ?)",
        default_users
    )
    conn.commit()

def get_db_connection():
    if DB_TYPE == "postgres" and psycopg2 is not None:
        try:
            conn = psycopg2.connect(
                host=PG_HOST, port=PG_PORT, database=PG_DB,
                user=PG_USER, password=PG_PASS
            )
            return DBConn(conn, is_pg=True)
        except Exception as e:
            print(f"PostgreSQL Error: {e}; falling back to local SQLite database.db")
    elif DB_TYPE == "postgres":
        print("PostgreSQL driver missing; falling back to local SQLite database.db")

    sqlite_name = DB_NAME if DB_TYPE != "postgres" else "database.db"
    if not sqlite_name.endswith(".db"):
        sqlite_name = "database.db"
    conn = sqlite3.connect(sqlite_name)
    conn.row_factory = sqlite3.Row
    # Local fallback mode can run under restricted Windows folders where SQLite
    # rollback journal files fail with disk I/O errors. Keep writes in-place.
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    ensure_sqlite_schema(conn)
    return DBConn(conn, is_pg=False)

def parse_db_json(field_data):
    """Safely parse JSON fields that might already be lists/dicts in Postgres."""
    if not field_data: return []
    if isinstance(field_data, str):
        try: return json.loads(field_data)
        except: return []
    return field_data

def clean_b64(b64_str):
    """Robustly strips prefixes and fixes padding for b64 strings."""
    if not b64_str: return ""
    b64_str = str(b64_str).strip()
    
    # Handle multiple prefixes (take the last part)
    if 'base64,' in b64_str:
        b64_str = b64_str.split('base64,')[-1]
    elif ',' in b64_str:
        b64_str = b64_str.split(',')[-1]
        
    # Remove any internal whitespace or problematic chars
    b64_str = "".join(b64_str.split())
    
    # Standardize URL-safe base64 to standard base64
    b64_str = b64_str.replace('-', '+').replace('_', '/')
    
    # Add padding if needed
    missing_padding = len(b64_str) % 4
    if missing_padding == 1:
        # One extra char is invalid in B64; discarding it to prevent crash
        b64_str = b64_str[:-1]
    elif missing_padding > 1:
        b64_str += '=' * (4 - missing_padding)
        
    return b64_str

def normalize_pole_id_text(text):
    if not text:
        logger.warning(f"[OCR] normalize rejected: {repr(text)}")
        return "Not Found"

    raw = str(text).upper()
    raw = raw.replace("|", "1").replace("!", "1")
    raw = raw.replace("}", "7").replace("]", "7").replace(")", "7")
    raw = raw.replace("{", "6").replace("[", "6").replace("(", "6")
    raw = raw.replace("₹", "").replace("¥", "").replace("`", "")
    spaced = re.sub(r"[^A-Z0-9]+", " ", raw).strip()
    compact = re.sub(r"[^A-Z0-9]", "", raw)
    if len(compact) < 3:
        logger.warning(f"[OCR] normalize rejected: {repr(text)}")
        return "Not Found"

    letters_view = spaced.replace("0", "O").replace("5", "S").replace("2", "Z").replace("6", "G").replace("8", "B")
    rdss_match = re.search(r"R[CDOG]SS?(?:[\s-]+)?([0-9OQSBZIL]{1,4})", letters_view)
    if rdss_match:
        digits_text = rdss_match.group(1)
        digits_text = digits_text.replace("O", "0").replace("Q", "0")
        digits_text = re.sub(r"(?<=\d)[IL]|(?<=^)[IL](?=\d)", "1", digits_text)
        digits_text = digits_text.replace("S", "5").replace("B", "8").replace("Z", "2")
        digits = re.search(r"\d{1,4}", digits_text)
        if digits:
            return f"RDSS {digits.group(0)}"
        return "RDSS"

    generic = re.search(r"\b([A-Z]{2,6})[\s-]*([0-9OQSBZIL]{1,4})\b", spaced)
    if generic:
        prefix = generic.group(1)
        if prefix in {"ID", "IV", "IMG", "IP", "IMAGE", "PAGE", "FIG", "TABLE"}:
            logger.warning(f"[OCR] normalize rejected: {repr(text)}")
            return "Not Found"
        digits_text = generic.group(2).replace("O", "0").replace("Q", "0")
        digits_text = re.sub(r"(?<=\d)[IL]|(?<=^)[IL](?=\d)", "1", digits_text)
        digits_text = digits_text.replace("S", "5").replace("B", "8").replace("Z", "2")
        digits = re.search(r"\d{1,4}", digits_text)
        if digits:
            return f"{prefix} {digits.group(0)}"
    logger.warning(f"[OCR] normalize rejected: {repr(text)}")
    return "Not Found"


def extract_ocr_text_lines(raw_text):
    if not raw_text:
        return []

    raw = str(raw_text).upper()
    spaced = re.sub(r"[^A-Z0-9]+", " ", raw).strip()
    if not spaced:
        return []

    lines = []
    seen = set()

    main_pole_id = normalize_pole_id_text(raw_text)
    if main_pole_id != "Not Found":
        lines.append(main_pole_id)
        seen.add(main_pole_id)

    for match in re.finditer(r"\b([A-Z]{2,6})[\s-]*([0-9OQSBZIL]{1,4})\b", spaced):
        prefix = match.group(1)
        digits_text = match.group(2).replace("O", "0").replace("Q", "0")
        digits_text = re.sub(r"(?<=\d)[IL]|(?<=^)[IL](?=\d)", "1", digits_text)
        digits_text = digits_text.replace("S", "5").replace("B", "8").replace("Z", "2")
        digits = re.search(r"\d{1,4}", digits_text)
        if not digits:
            continue
        if prefix == "RDSS":
            candidate = f"RDSS {digits.group(0)}"
        else:
            candidate = f"{prefix}{digits.group(0)}"
        if normalize_pole_id_text(candidate) == main_pole_id:
            continue
        if candidate not in seen:
            lines.append(candidate)
            seen.add(candidate)

    return lines

def pole_id_score(text):
    normalized = normalize_pole_id_text(text)
    if normalized == "Not Found":
        return -9999
    score = 1000
    if normalized.startswith("RDSS"):
        score += 500
    digit_groups = re.findall(r"\d+", normalized)
    if digit_groups:
        longest_group = max(len(group) for group in digit_groups)
        score += sum(len(group) for group in digit_groups) * 100
        if longest_group >= 2:
            score += 300
    return score


def get_rapidocr_engine():
    global rapidocr_engine
    if rapidocr_engine is not None:
        return rapidocr_engine
    try:
        from rapidocr import RapidOCR
    except ImportError:
        from rapidocr_onnxruntime import RapidOCR
    rapidocr_engine = RapidOCR()
    return rapidocr_engine

def find_pole_tag_crop(img):
    try:
        import cv2
        import numpy as np

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        if h < 20 or w < 20:
            return img

        def _best_dark_box(region, x_offset=0):
            region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            region_h, region_w = region_gray.shape[:2]
            mean_val = float(np.mean(region_gray))
            thresh_val = min(105, max(30, int(mean_val * 0.72)))
            _, dark_mask = cv2.threshold(region_gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel, iterations=1)

            contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best_box = None
            best_score = 0.0

            for c in contours:
                area = cv2.contourArea(c)
                if area < max(180, (region_h * region_w) * 0.0025):
                    continue
                x, y, bw, bh = cv2.boundingRect(c)
                if bw < 18 or bh < 24:
                    continue
                if bw > region_w * 0.75 or bh > region_h * 0.65:
                    continue
                if x <= 2 or y <= 2 or (x + bw) >= (region_w - 2) or (y + bh) >= (region_h - 2):
                    continue

                rect_area = float(bw * bh)
                fill_ratio = area / rect_area if rect_area else 0.0
                aspect_ratio = bw / float(max(bh, 1))
                center_x = (x + bw / 2.0) / float(region_w)
                center_y = (y + bh / 2.0) / float(region_h)

                score = area * max(fill_ratio, 0.45)
                if 0.18 <= aspect_ratio <= 1.25:
                    score *= 1.25
                elif aspect_ratio > 1.8:
                    score *= 0.65

                if 0.18 <= center_x <= 0.72:
                    score *= 1.35
                elif center_x >= 0.82:
                    score *= 0.2
                else:
                    score *= 0.8

                if 0.18 <= center_y <= 0.88:
                    score *= 1.1
                if fill_ratio < 0.22:
                    score *= 0.5

                if score > best_score:
                    best_score = score
                    best_box = (x + x_offset, y, bw, bh)
            return best_box

        best_box = _best_dark_box(img)
        if not best_box:
            band_x1 = max(0, int(w * 0.18))
            band_x2 = min(w, int(w * 0.72))
            best_box = _best_dark_box(img[:, band_x1:band_x2], x_offset=band_x1)

        if not best_box:
            return img

        x, y, bw, bh = best_box
        pad_x = max(24, int(bw * 0.18))
        pad_top = max(32, int(bh * 0.30))
        pad_bottom = max(40, int(bh * 0.26))
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_top)
        x2 = min(w, x + bw + pad_x)
        y2 = min(h, y + bh + pad_bottom)
        crop = img[y1:y2, x1:x2]
        return crop if crop.size else img
    except Exception as exc:
        print(f"[OCR] Local OCR crop detection error: {exc}")
        return img

def prepare_ocr_image(img):
    try:
        import cv2
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        prepared = cv2.bitwise_not(thresh)
        return cv2.cvtColor(prepared, cv2.COLOR_GRAY2BGR)
    except Exception as exc:
        print(f"[OCR] OCR preprocessing error: {exc}")
        return img

def read_pole_id_for_blob(blob):
    temp_filename = os.path.join(UPLOADS_FOLDER, f"temp_ocr_{uuid.uuid4()}.jpg")
    try:
        with open(temp_filename, "wb") as f:
            f.write(blob)
        return read_pole_id_with_rapidocr(temp_filename)
    finally:
        safe_remove_file(temp_filename, "OCR temp image")

def read_pole_id_with_rapidocr(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return "Not Found"

        crop_raw = find_pole_tag_crop(img)
        crop_prepped = prepare_ocr_image(crop_raw)
        print("[OCR] Running RapidOCR on pole tag crop...")
        engine = get_rapidocr_engine()
        best_payload = {
            "pole_id": "Not Found",
            "text_lines": [],
            "raw_text": "",
        }
        best_score = -9999

        crop_variants = (("raw", crop_raw), ("prep", crop_prepped))
        saw_any_crop_text = False

        for variant_name, candidate_img in crop_variants:
            result, _ = engine(candidate_img)
            pieces = []
            if result:
                for line in result:
                    if not line or len(line) < 2:
                        continue
                    text = str(line[1]).strip()
                    if text:
                        pieces.append(text)
            clean_text = " ".join(pieces).strip()
            print(f"[OCR] RapidOCR raw output ({variant_name}): {repr(clean_text)}")
            if clean_text:
                saw_any_crop_text = True
            normalized = normalize_pole_id_text(clean_text)
            score = pole_id_score(normalized)
            if score > best_score:
                best_score = score
                best_payload = {
                    "pole_id": normalized,
                    "text_lines": extract_ocr_text_lines(clean_text),
                    "raw_text": clean_text,
                }

        if not saw_any_crop_text:
            print("[OCR] Crop OCR empty, retrying RapidOCR on full image...")
            full_prepped = prepare_ocr_image(img)
            for variant_name, candidate_img in (("full_raw", img), ("full_prep", full_prepped)):
                result, _ = engine(candidate_img)
                pieces = []
                if result:
                    for line in result:
                        if not line or len(line) < 2:
                            continue
                        text = str(line[1]).strip()
                        if text:
                            pieces.append(text)
                clean_text = " ".join(pieces).strip()
                print(f"[OCR] RapidOCR raw output ({variant_name}): {repr(clean_text)}")
                normalized = normalize_pole_id_text(clean_text)
                score = pole_id_score(normalized)
                if score > best_score:
                    best_score = score
                    best_payload = {
                        "pole_id": normalized,
                        "text_lines": extract_ocr_text_lines(clean_text),
                        "raw_text": clean_text,
                    }

        if best_payload["pole_id"] == "Not Found":
            _save_debug_crop(crop_raw, tag="ocr_fail_raw")
            _save_debug_crop(crop_prepped, tag="ocr_fail_prep")
            return best_payload

        print(f"[OCR] RapidOCR selected: '{best_payload['pole_id']}' from '{best_payload['raw_text']}'")
        return best_payload
    except Exception as exc:
        print(f"[OCR] RapidOCR error: {exc}")
        return {
            "pole_id": "Not Found",
            "text_lines": [],
            "raw_text": "",
        }

# =========================
def log_activity(user, action, details=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # DEEP LOGGING: Identify exactly what is being sent to the DB
    print(f"[DB_LOG] user={user} ({type(user)}), action={action} ({type(action)}), details={details} ({type(details)})")
    
    # Defensive casting
    u = str(user) if user is not None else "system"
    a = str(action) if action is not None else "unknown"
    d = json.dumps(details) if isinstance(details, (dict, list)) else (str(details) if details is not None else "")

    conn = get_db_connection()
    try:
        conn.execute('INSERT INTO activity_logs (user_name, action, details, timestamp) VALUES (?, ?, ?, ?)',
                     (u, a, d, timestamp))
        conn.commit()
    except Exception as e:
        print(f"[DB_ERROR] log_activity failed: {e}")
    finally:
        conn.close()

def get_ngrok_url():
    # Ngrok polling removed per user request
    return None

# =========================
# AUTHENTICATION
# =========================
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            if request.path.startswith('/api/'):
                return jsonify({"status": "error", "message": "Authentication required"}), 401
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session or session['role'] != 'admin':
            return redirect(url_for('home'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()
        
        if user and check_password_hash(user['password'], password):
            session['user'] = user['username']
            session['role'] = user['role']
            log_activity(username, "login", f"Role: {user['role']}")
            if user['role'] == 'admin':
                return redirect(url_for('admin_dashboard'))
            return redirect(url_for('home'))
        
        return render_template('login.html', error="Invalid credentials", ngrok_url=get_ngrok_url())
    
    return render_template('login.html', ngrok_url=get_ngrok_url())

@app.route('/api/login', methods=['POST'])
def api_login():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400
            
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()
        
        if user and check_password_hash(user['password'], password):
            session['user'] = user['username']
            session['role'] = user['role']
            log_activity(username, "api_login", f"Role: {user['role']}")
            redirect_url = url_for('admin_dashboard') if user['role'] == 'admin' else url_for('home')
            return jsonify({"status": "success", "redirect": redirect_url})
            
        return jsonify({"status": "error", "message": "Invalid credentials"}), 401
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Server Error: {str(e)}"}), 500

def sanitize_database():
    """Permanent fix: Strips all legacy prefixes from the DB so clean_b64 never fails."""
    conn = get_db_connection()
    try:
        rows = conn.execute('SELECT id, image_b64 FROM asset_images').fetchall()
        count = 0
        for r in rows:
            b = r['image_b64']
            if b and (',' in b or 'base64' in b):
                # Take only the standard b64 part
                cleaned = b.split(',')[-1].split('base64,').pop().strip()
                conn.execute('UPDATE asset_images SET image_b64 = ? WHERE id = ?', (cleaned, r['id']))
                count += 1
        conn.commit()
        if count > 0:
            print(f"--- [DATABASE SANITIZER] Cleaned {count} malformed image rows ---")
    except Exception as e:
        print(f"Sanitizer Error: {e}")
    finally:
        conn.close()

# Run cleanup on start
with app.app_context():
    try:
        sanitize_database()
    except Exception as e:
        print(f"Sanitizer skipped: {e}")

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()

        if not username or not password:
            return render_template('signup.html', error="All fields are required")
        if password != confirm_password:
            return render_template('signup.html', error="Passwords do not match")

        conn = get_db_connection()
        existing = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        if existing:
            conn.close()
            return render_template('signup.html', error="Username already exists")

        hashed_pw = generate_password_hash(password)
        conn.execute('INSERT INTO users (username, password, role) VALUES (?, ?, ?)', (username, hashed_pw, 'user'))
        conn.commit()
        
        # Log them in automatically
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()
        
        session['user'] = user['username']
        session['role'] = user['role']
        log_activity(username, "signup", "New user registered via web UI")
        
        return redirect(url_for('home'))

    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# =========================
# ADMIN USER MANAGEMENT
# =========================
@app.route('/api/admin/users', methods=['GET'])
@admin_required
def get_all_users():
    conn = get_db_connection()
    users = conn.execute('SELECT id, username, role FROM users').fetchall()
    conn.close()
    return jsonify([dict(u) for u in users])

@app.route('/api/admin/users/<username>', methods=['DELETE'])
@admin_required
def delete_user(username):
    # Prevent self-deletion
    if username == session.get('user'):
        return jsonify({'status': 'error', 'message': 'Cannot delete active session user'}), 400
        
    conn = get_db_connection()
    cursor = conn.execute('DELETE FROM users WHERE username = ?', (username,))
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()
    
    if deleted:
        log_activity(session.get('user', 'admin'), "delete_user", f"Deleted user: {username}")
        return jsonify({'status': 'success'})
    else:
        return jsonify({'status': 'error', 'message': 'User not found'}), 404

# =========================
# IMAGE PROCESSING
# =========================
def process_image_file(file_stream, fast_mode=False, enable_ocr=True):
    """
    Main diagnostic entry point.
    Combines Rule Engine (InfrastructurePipeline) with UNet Conductor Segmentation.
    """
    load_detection_models(load_unet=not fast_mode)

    # Create a temporary file to run the pipeline.predict (which expects a path)
    import gc
    import psutil
    import cv2
    import numpy as np
    import torch
    
    def log_mem(step):
        m = psutil.Process().memory_info().rss / (1024 * 1024)
        print(f"[Memory] {step}: {m:.1f} MB")

    log_mem("Start Inference")
    t0 = time.perf_counter()
    temp_filename = os.path.join(UPLOADS_FOLDER, f"temp_{uuid.uuid4()}.jpg")
    try:
        with open(temp_filename, "wb") as f:
            f.write(file_stream.read())

        # Run model inference and a single OCR API call in parallel.
        log_mem("Before Pipeline")
        if enable_ocr:
            with ThreadPoolExecutor(max_workers=2) as executor:
                pipe_future = executor.submit(pipeline_engine.predict, temp_filename, False, None, fast_mode)
                ocr_future = executor.submit(read_pole_id_with_rapidocr, temp_filename)
                pipe_res = pipe_future.result()
                ocr_payload = ocr_future.result()
        else:
            pipe_res = pipeline_engine.predict(temp_filename, visualize=False, fast_mode=fast_mode)
            ocr_payload = {"pole_id": "Not Found", "text_lines": [], "raw_text": ""}
        log_mem("After Pipeline")
        print(f"[Timing] Pipeline: {time.perf_counter() - t0:.2f}s")
        gc.collect()

        # Reload image for UNet processing and base64 response
        img = cv2.imread(temp_filename)
        h, w = img.shape[:2]
        
        if not fast_mode:
            # 2. Process Conductors with UNet Segmentation Model (ResNet34)
            input_img = cv2.resize(img, (512, 512)).transpose(2, 0, 1) / 255.0
            tensor = torch.tensor(input_img[None, ...], dtype=torch.float32).to(device)
            
            with torch.no_grad():
                out = unet_model(tensor)
                mask = torch.sigmoid(out).squeeze().cpu().numpy()
                log_mem("After UNet")
                del out
            
            del tensor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            mask_binary = (mask > 0.85).astype(np.uint8) * 255
            mask_resized = cv2.resize(mask_binary, (w, h), interpolation=cv2.INTER_NEAREST)

            # Thickness Measurement via Distance Transform & Skeletonize
            dist = cv2.distanceTransform(mask_resized, cv2.DIST_L2, 5)
            from skimage.morphology import skeletonize
            skel = (skeletonize(mask_resized / 255.0) > 0).astype(np.uint8)

            # Bridge gaps for continuous polygons (Wider kernel to fix wire count)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
            mask_closed = cv2.morphologyEx(mask_resized, cv2.MORPH_CLOSE, kernel)
        
        # 3. Create Hardware Blackout Mask (Wire Detection Last)
        # Prevents wires from "hallucinating" over insulators/poles
        hardware_mask = np.zeros((h, w), dtype=np.uint8)
        
        final_detections = []

        # A. Map Rule Engine Components to UI format (with OBB Polygons)
        # Each component in pipe_res is now (box, conf, angle, polygon)
        for ins in pipe_res.insulators:
            if float(ins.detection_conf) < 0.40:
                continue
            insulator_label = getattr(ins, "detector_class", None) or f"INS_{ins.type_final}".upper()
            # Map to Hardware Mask with 5px buffer
            x1, y1, x2, y2 = [int(v) for v in ins.box]
            cv2.rectangle(hardware_mask, (max(0, x1-5), max(0, y1-5)), (min(w, x2+5), min(h, y2+5)), 255, -1)
            
            final_detections.append({
                "label": insulator_label,
                "confidence": float(ins.detection_conf),
                "bbox": [int(x) for x in ins.box],
                "polygon": ins.obb_polygon if hasattr(ins, 'obb_polygon') else None,
                "source": MODEL_PATHS["insulator"],
                "details": {
                    "detector_class": insulator_label,
                    "voltage": ins.voltage,
                    "shed_count": int(ins.shed_count),
                    "type": ins.type_final
                }
            })
        
        for ca in pipe_res.all_arms:
            if float(ca.detection_conf) < 0.40:
                continue
            # Map to Hardware Mask with 5px buffer
            x1, y1, x2, y2 = [int(v) for v in ca.box]
            cv2.rectangle(hardware_mask, (max(0, x1-5), max(0, y1-5)), (min(w, x2+5), min(h, y2+5)), 255, -1)
            
            final_detections.append({
                "label": ca.pole_type,
                "confidence": float(ca.detection_conf),
                "bbox": [int(x) for x in ca.box],
                "polygon": ca.obb_polygon if hasattr(ca, 'obb_polygon') else None,
                "source": MODEL_PATHS["components"],
                "details": {
                    "shape": ca.shape
                }
            })
        
        for po in pipe_res.all_poles:
            if float(po.detection_conf) < 0.40:
                continue
            # Map to Hardware Mask with 5px buffer
            x1, y1, x2, y2 = [int(v) for v in po.box]
            cv2.rectangle(hardware_mask, (max(0, x1-5), max(0, y1-5)), (min(w, x2+5), min(h, y2+5)), 255, -1)
            
            final_detections.append({
                "label": "strut_pole" if po.pole_type == "strut_pole" else "pole",
                "confidence": float(po.detection_conf),
                "bbox": [int(x) for x in po.box],
                "polygon": po.obb_polygon if hasattr(po, 'obb_polygon') else None,
                "source": MODEL_PATHS["pole"],
                "details": {
                    "type": po.pole_type,
                    "lean": round(float(po.lean_angle_deg), 1)
                }
            })

        for box, conf, poly in pipe_res.street_lights:
            if float(conf) < image_display_threshold("street_light"):
                continue
            # Map to Hardware Mask (Street lights are hardware too, wires shouldn't pass THROUGH them)
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(hardware_mask, (max(0, x1-5), max(0, y1-5)), (min(w, x2+5), min(h, y2+5)), 255, -1)
            
            final_detections.append({
                "label": "street_light",
                "confidence": float(conf),
                "bbox": [int(x) for x in box],
                "polygon": poly,
                "source": MODEL_PATHS["components"],
                "details": {"type": "Standard Lamp"}
            })

        for label, box, conf, poly in pipe_res.others:
            if float(conf) < image_display_threshold(label):
                continue
            # Add to hardware exclusion mask
            bw, bh = box[2]-box[0], box[3]-box[1]
            if bw > 100 or bh > 100:
                cv2.rectangle(hardware_mask, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), 255, -1)
            
            # Add back to UI detections
            final_detections.append({
                "label": label.upper(),
                "confidence": float(conf),
                "bbox": [int(x) for x in box],
                "polygon": poly,
                "source": MODEL_PATHS["components"],
                "details": {"type": "Hardware"}
            })

        if not fast_mode:
            # --- Wire Discovery Phase 2: Exclude static hardware ---
            mask_final = cv2.bitwise_and(mask_closed, cv2.bitwise_not(hardware_mask))
            
            # B. Generate Conductor Polygons from clean mask
            contours, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                cx, cy, cw, ch = cv2.boundingRect(c)
                area = cv2.contourArea(c)
                
                if area < 500 or cw + ch < 80 or area < 100: 
                    continue
                
                aspect_ratio = max(cw, ch) / max(1, min(cw, ch))
                solidity = area / (cw * ch)
                if cw > 150 and ch > 150 and solidity > 0.5 and aspect_ratio < 1.8:
                    continue
                
                epsilon = 0.01 * cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, epsilon, True)
                polygon = [[int(pt[0][0]), int(pt[0][1])] for pt in approx]
                
                c_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(c_mask, [c], -1, 255, -1)
                local_skel = skel & (c_mask > 0)
                local_thickness = dist[local_skel > 0] * 2
                avg_thick = float(np.mean(local_thickness)) if len(local_thickness) > 0 else 0.0
                
                if avg_thick > 80:
                    continue
                
                final_detections.append({
                    "label": "conductor",
                    "confidence": 0.90,
                    "bbox": [cx, cy, cx+cw, cy+ch],
                    "polygon": polygon,
                    "source": MODEL_PATHS["conductor_unet"],
                    "thickness": round(avg_thick, 1)
                })

        # Prepare Master Data (Asset Identity)
        master_data = {
            "final_class": pipe_res.final_class,
            "voltage": pipe_res.voltage,
            "pole_id": ocr_payload["pole_id"] if ocr_payload["pole_id"] != "Not Found" else pipe_res.pole_id,
            "ocr_text_lines": ocr_payload["text_lines"],
            "reason": pipe_res.reason,
            "confidence": pipe_res.confidence,
            "pole_lean_angle": pipe_res.pole_orientation.lean_angle_deg if pipe_res.pole_orientation else 0.0,
            "pole_type": pipe_res.pole_orientation.pole_type if pipe_res.pole_orientation else "none",
            "pole_status": pipe_res.pole_orientation.fault_severity if pipe_res.pole_orientation else "none",
            "model_summary": {
                "structural": MODEL_PATHS["pole"],
                "components": MODEL_PATHS["components"],
                "insulator": MODEL_PATHS["insulator"],
                "segmentation": "skipped in fast mode" if fast_mode else MODEL_PATHS["conductor_unet"]
            }
        }

        # D. Map to Survey Questionnaire (for external integration)
        has_strut = any(po.pole_type == "strut_pole" for po in pipe_res.all_poles)
        survey_q = {
            "strut_pole": "Yes" if has_strut else "No",
            "strut_pole_count": sum(1 for po in pipe_res.all_poles if po.pole_type == "strut_pole"),
            "tilt_angle": f"{pipe_res.pole_orientation.lean_angle_deg:.1f}" if pipe_res.pole_orientation else "0.0",
            "is_leaning": "Yes" if (pipe_res.pole_orientation and pipe_res.pole_orientation.lean_angle_deg > 5.0) else "No",
            "vegetation": "Yes" if pipe_res.flags.get("has_vegetation") else "No"
        }

        # Encode for response
        import base64 as _base64
        _, buffer = cv2.imencode('.jpg', img)
        img_b64 = _base64.b64encode(buffer).decode('utf-8')
        final_counts = Counter(str(d.get("label", "")).upper() for d in final_detections)
        print(f"[IMAGE-RESULT] Final UI detections count={len(final_detections)} classes={dict(final_counts)}", flush=True)
        for det in final_detections:
            label = str(det.get("label", "")).upper()
            if label in {"SPECIAL_CLAMP", "STREET_LIGHT"}:
                print(
                    "[IMAGE-RESULT] "
                    f"ui_label={label} conf={float(det.get('confidence', 0.0)):.3f} "
                    f"bbox={det.get('bbox')} source={det.get('source')}",
                    flush=True
                )

        return {
            "detections": final_detections,
            "master": master_data,
            "survey_questionnaire": survey_q,
            "annotated_image": img_b64,
            "width": w,
            "height": h
        }
    finally:
        print(f"[Timing] Total request: {time.perf_counter() - t0:.2f}s")
        # Final safety cleanup for 1.7GB RAM environment
        if 'img' in locals(): del img
        if 'pipe_res' in locals(): del pipe_res
        if 'mask' in locals(): del mask
        if 'tensor' in locals(): del tensor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Cleanup temporary image file. Do not fail inference if Windows keeps
        # the file locked briefly after OpenCV/Ultralytics reads it.
        safe_remove_file(temp_filename, "inference temp image")
        log_mem("Request End")

# =========================
# FLASK ROUTES
# =========================
@app.route('/predict_stream', methods=['POST'])
@login_required
def predict_stream():
    """Lightweight endpoint for AR Camera Stream."""
    data = request.json
    if not data or 'image' not in data:
        return jsonify({"error": "No image payload"}), 400
    
    img_b64 = data['image'].split(',').pop() if ',' in data['image'] else data['image']
    img_data = base64.b64decode(img_b64)
    file_stream = io.BytesIO(img_data)
    
    try:
        result = process_image_file(file_stream, fast_mode=True)
        # Strip annotated image to save bandwidth for the stream
        if "annotated_image" in result:
            del result["annotated_image"]
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Inference Error: {str(e)}"}), 500

@app.route('/')
@login_required
def home():
    ngrok_url = get_ngrok_url()
    return render_template('index.html', ngrok_url=ngrok_url, role=session.get('role', 'user'))

@app.route('/admin')
@admin_required
def admin_dashboard():
    return render_template('admin.html')

@app.route('/admin/export/global/excel')
@admin_required
def export_global_excel():
    _, _, generate_global_excel, _ = load_report_generators()
    conn = get_db_connection()
    assets = conn.execute('SELECT * FROM assets ORDER BY timestamp DESC').fetchall()
    asset_images = conn.execute('SELECT * FROM asset_images').fetchall()
    conn.close()

    # Group images by asset
    img_map = {}
    for img in asset_images:
        aid = img['asset_id']
        if aid not in img_map: img_map[aid] = []
        parsed_img = dict(img)
        parsed_img['detections'] = parse_db_json(img['detections'])
        img_map[aid].append(parsed_img)

    assets_list = []
    for a in assets:
        a_dict = dict(a)
        a_dict['images'] = img_map.get(a['id'], [])
        assets_list.append(a_dict)

    excel_buffer = generate_global_excel(assets_list)
    filename = f"Global_Inspection_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
    return send_file(excel_buffer, download_name=filename, as_attachment=True, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

@app.route('/admin/export/global/pdf')
@admin_required
def export_global_pdf():
    _, _, _, generate_global_pdf = load_report_generators()
    conn = get_db_connection()
    assets = conn.execute('SELECT * FROM assets ORDER BY timestamp DESC').fetchall()
    conn.close()

    assets_list = [dict(a) for a in assets]
    pdf_buffer = generate_global_pdf(assets_list)
    filename = f"Global_Inspection_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    return send_file(pdf_buffer, download_name=filename, as_attachment=True, mimetype='application/pdf')

@app.route('/admin/asset/<asset_id>')
@admin_required
def view_asset(asset_id):
    conn = get_db_connection()
    asset_row = conn.execute('SELECT * FROM assets WHERE id = ?', (asset_id,)).fetchone()
    
    if not asset_row:
        conn.close()
        return "Asset not found", 404
        
    image_rows = conn.execute('SELECT * FROM asset_images WHERE asset_id = ?', (asset_id,)).fetchall()
    conn.close()
    
    asset = dict(asset_row)
    images = []
    total_detections = 0

    for row in image_rows:
        img_dict = dict(row)
        img_dict['detections'] = parse_db_json(img_dict['detections'])
        total_detections += len(img_dict['detections'])
        images.append(img_dict)
    
    asset['images'] = images
    asset['total_count'] = total_detections
    
    return render_template('asset_detail.html', asset=asset)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    # Handle up to 3 images
    files = []
    for i in range(1, 4):
        key = f'image{i}'
        if key in request.files:
            files.append(request.files[key])
    
    # Fallback for single 'image' key
    if not files and 'image' in request.files:
        files.append(request.files['image'])

    if not files:
        return jsonify({"error": "No images uploaded"}), 400

    try:
        if len(files) == 1:
            # Single image: fast processing for interactive worker review
            result = process_image_file(files[0], fast_mode=True)
            return jsonify(result)
        else:
            # Multi-image: Merged processing
            print(f"DEBUG: Processing {len(files)} images in merged mode...")
            return jsonify(process_multi_images(files))
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Inference Error: {str(e)}"}), 500

def process_multi_images(file_streams):
    """
    Processes multiple images sequentially to avoid Out of Memory (OOM) on 1.7GB RAM instances.
    """
    results = [None] * len(file_streams)
    file_blobs = []
    for idx, stream in enumerate(file_streams):
        try:
            print(f"DEBUG: Processing image {idx+1}/{len(file_streams)} in merged mode...")
            blob = stream.read()
            file_blobs.append(blob)
            res = process_image_file(io.BytesIO(blob), fast_mode=True, enable_ocr=False)
            if res:
                results[idx] = res
        except Exception as e:
            print(f"[ERROR] Failed to process image {idx+1} in merged mode: {e}")
            import traceback
            traceback.print_exc()
            file_blobs.append(None)

    valid_results = [r for r in results if r is not None]
    if not valid_results:
        return {"error": "No valid results generated from uploaded images"}

    # --- MERGE LOGIC ---
    ocr_results = [{"pole_id": "Not Found", "text_lines": [], "raw_text": ""} for _ in results]

    # 1. Choose the 'Best' result as the Master
    best_result = max(valid_results, key=lambda x: (len(x.get('detections', [])), x['master']['confidence'] == 'high'))
    
    # Clone to avoid modifying the original element inside `results` directly
    merged_result = {
        "detections": best_result.get("detections", []),
        "master": {**best_result.get("master", {})},
        "survey_questionnaire": best_result.get("survey_questionnaire", {}),
        "annotated_image": best_result.get("annotated_image", ""),
        "width": best_result.get("width", 0),
        "height": best_result.get("height", 0)
    }

    for idx, blob in enumerate(file_blobs):
        if not blob or results[idx] is None:
            continue
        print(f"DEBUG: Running OCR on image {idx+1}/{len(file_streams)} in merged mode...")
        try:
            ocr_results[idx] = read_pole_id_for_blob(blob)
        except Exception as e:
            print(f"[ERROR] Merged OCR failed on image {idx+1}: {e}")

    best_ocr = max(ocr_results, key=lambda x: pole_id_score(x.get("pole_id"))) if ocr_results else {"pole_id": "Not Found", "text_lines": []}

    merged_result['master']['pole_id'] = best_ocr.get("pole_id", "Not Found")
    merged_result['master']['ocr_text_lines'] = best_ocr.get("text_lines", [])

    # Add a flag that this was a merged result
    merged_result['master']['reason'] = f"[Merged from {len(valid_results)} images] " + merged_result['master']['reason']
    
    # Store all images in the result so the UI can display them maintaining correct index mapping
    merged_result['all_images'] = [r['annotated_image'] if r else None for r in results]
    merged_result['all_detections'] = [r['detections'] if r else [] for r in results]
    merged_result['all_dims'] = [{"width": r['width'], "height": r['height']} if r else {"width": 0, "height": 0} for r in results]
    merged_result['all_pole_ids'] = [
        ocr_results[idx].get("pole_id", "Not Found") if r else 'Not Found'
        for idx, r in enumerate(results)
    ]
    merged_result['all_ocr_text_lines'] = [
        ocr_results[idx].get("text_lines", []) if r else []
        for idx, r in enumerate(results)
    ]
    return merged_result

def _normalise_video_pole_label(raw_label):
    label = str(raw_label or "").strip().lower().replace(" ", "_").replace("-", "_")
    if "strut" in label:
        return "STRUT_POLE"
    if "pole" in label:
        return "MAIN_POLE"
    return label.upper()

def _draw_video_detection(frame, bbox, label, confidence, track_id=None):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    color = (80, 180, 255) if label == "MAIN_POLE" else (90, 220, 120)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    title = f"{label.replace('_', ' ')} {confidence:.2f}"
    if track_id is not None:
        title = f"ID {track_id} | {title}"

    (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    y_text = max(24, y1 - 8)
    cv2.rectangle(frame, (x1, y_text - th - 8), (x1 + tw + 10, y_text + 4), color, -1)
    cv2.putText(frame, title, (x1 + 5, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (5, 8, 14), 2, cv2.LINE_AA)

def _frame_sharpness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def _clip_bbox(bbox, width, height, padding=0):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return [
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(width, x2 + padding),
        min(height, y2 + padding),
    ]

def _scale_bbox(bbox, scale_x, scale_y):
    x1, y1, x2, y2 = bbox
    return [
        int(round(x1 * scale_x)),
        int(round(y1 * scale_y)),
        int(round(x2 * scale_x)),
        int(round(y2 * scale_y)),
    ]

def _bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / max(1, area_a + area_b - inter)

def _center_distance_ratio(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
    bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
    diag = max(1.0, (((ax2 - ax1) + (bx2 - bx1)) / 2) ** 2 + (((ay2 - ay1) + (by2 - by1)) / 2) ** 2)
    return (((acx - bcx) ** 2 + (acy - bcy) ** 2) / diag) ** 0.5

def _vertical_overlap_ratio(a, b):
    _, ay1, _, ay2 = a
    _, by1, _, by2 = b
    overlap = max(0, min(ay2, by2) - max(ay1, by1))
    return overlap / max(1, min(ay2 - ay1, by2 - by1))

def _bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0

def _bbox_height(bbox):
    return max(1, int(bbox[3]) - int(bbox[1]))

def _video_best_frame_score(confidence, bbox, frame_width, frame_height):
    cx, _ = _bbox_center(bbox)
    half_width = max(1.0, frame_width / 2.0)
    horizontal_centrality = 1.0 - min(1.0, abs(cx - half_width) / half_width)
    height_ratio = min(1.0, _bbox_height(bbox) / max(1.0, frame_height))
    return (float(confidence) * 0.5) + (horizontal_centrality * 0.3) + (height_ratio * 0.2)

def _track_best_time(track):
    best = track.get("best") or {}
    return float(best.get("frame_time", 0.0) or 0.0)

def _track_frame_indices(track):
    observations = track.get("observations") or {}
    return sorted(int(idx) for idx in observations.keys())

def _track_start_frame(track):
    frames = _track_frame_indices(track)
    return frames[0] if frames else int(track.get("start_frame", 0) or 0)

def _track_end_frame(track):
    frames = _track_frame_indices(track)
    return frames[-1] if frames else int(track.get("end_frame", 0) or 0)

def _track_observation_at(track, frame_index):
    observations = track.get("observations") or {}
    return observations.get(frame_index) or observations.get(str(frame_index))

def _track_edge_bbox(track, which="end"):
    frames = _track_frame_indices(track)
    if not frames:
        return (track.get("best") or {}).get("bbox", [0, 0, 0, 0])
    frame_index = frames[-1] if which == "end" else frames[0]
    obs = _track_observation_at(track, frame_index) or {}
    return obs.get("bbox", (track.get("best") or {}).get("bbox", [0, 0, 0, 0]))

def _track_centroid_distance_px(a, b):
    acx, acy = _bbox_center(a)
    bcx, bcy = _bbox_center(b)
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5

def _refresh_track_summary(track):
    observations = track.get("observations") or {}
    if not observations:
        return track
    best = max(observations.values(), key=lambda obs: float(obs.get("score", 0.0)))
    frames = sorted(int(idx) for idx in observations.keys())
    track["appearances"] = len(frames)
    track["start_frame"] = frames[0]
    track["end_frame"] = frames[-1]
    track["first_time"] = float(observations[frames[0]].get("frame_time", 0.0))
    track["last_time"] = float(observations[frames[-1]].get("frame_time", 0.0))
    track["best_score"] = float(best.get("score", 0.0))
    track["best"] = best
    return track

def _add_track_observation(track, frame_index, observation):
    observations = track.setdefault("observations", {})
    existing = observations.get(frame_index)
    if existing is None or float(observation.get("confidence", 0.0)) > float(existing.get("confidence", 0.0)):
        observations[frame_index] = observation
    return _refresh_track_summary(track)

def _time_gap_to_group(track, group):
    gap = _track_start_frame(track) - _track_end_frame(group)
    return max(0, gap)

def _same_physical_pole(track, group, frame_width=None, temporal_gap_max_frames=15):
    if track["label"] != group["label"]:
        return False
    track_id = track.get("track_id")
    if track_id is not None and track_id in group.get("track_ids", []):
        return True
    temporal_gap = _time_gap_to_group(track, group)
    if temporal_gap >= temporal_gap_max_frames:
        return False
    a = _track_edge_bbox(group, "end")
    b = _track_edge_bbox(track, "start")
    width_threshold = max(1.0, (frame_width or max(a[2], b[2], 1)) * 0.15)
    return _track_centroid_distance_px(a, b) < width_threshold

def _merge_pole_track_fragments(tracks, frame_width, temporal_gap_max_frames=15):
    groups = []
    fragments = [t for t in tracks.values() if t.get("best")]
    fragments.sort(key=lambda t: (t["label"], _track_start_frame(t), _track_end_frame(t), -t["best_score"]))

    for track in fragments:
        matched = None
        for group in groups:
            if _same_physical_pole(track, group, frame_width, temporal_gap_max_frames):
                matched = group
                break

        if matched is None:
            groups.append({
                "label": track["label"],
                "track_ids": [track["track_id"]] if track.get("track_id") is not None else [],
                "observations": dict(track.get("observations") or {}),
                "appearances": int(track.get("appearances", 0)),
                "best_score": float(track.get("best_score", 0.0)),
                "best": track.get("best"),
                "fragments": 1,
                "first_time": float(track.get("first_time", _track_best_time(track))),
                "last_time": float(track.get("last_time", _track_best_time(track))),
                "start_frame": _track_start_frame(track),
                "end_frame": _track_end_frame(track),
            })
            _refresh_track_summary(groups[-1])
            continue

        _merge_group_into(matched, track)

    return groups

def _merge_group_into(target, source):
    for track_id in source.get("track_ids", []):
        if track_id is not None and track_id not in target["track_ids"]:
            target["track_ids"].append(track_id)
    if source.get("track_id") is not None and source.get("track_id") not in target["track_ids"]:
        target["track_ids"].append(source["track_id"])
    target_observations = target.setdefault("observations", {})
    for frame_index, observation in (source.get("observations") or {}).items():
        existing = target_observations.get(frame_index)
        if existing is None or float(observation.get("confidence", 0.0)) > float(existing.get("confidence", 0.0)):
            target_observations[frame_index] = observation
    target["fragments"] += source.get("fragments", 1)
    return _refresh_track_summary(target)

def _merge_temporal_pole_events(groups, frame_width, max_gap_frames=15):
    """Join fallback tracker splits that are the same pole across adjacent time windows."""
    merged = []
    ordered = sorted(
        groups,
        key=lambda g: (g["label"], _track_start_frame(g), _track_end_frame(g))
    )

    for group in ordered:
        if not merged:
            merged.append(group)
            continue

        current = merged[-1]
        if current["label"] != group["label"]:
            merged.append(group)
            continue

        gap = _track_start_frame(group) - _track_end_frame(current)
        should_merge = 0 <= gap < max_gap_frames and _same_physical_pole(group, current, frame_width)

        if should_merge:
            _merge_group_into(current, group)
        else:
            merged.append(group)

    return merged

def _video_tracker_backend():
    try:
        import lap  # noqa: F401
        return "bytetrack"
    except Exception:
        return "fallback"

def process_video_path(input_path, trim_start=0.0, trim_duration=30.0, job_id=None, pole_model=None, worker_name=None):
    """
    Process a selected 30-second video segment:
    configurable FPS sampling -> pole detection -> ByteTrack -> one track per pole ->
    best frame selection. Video mode only returns MAIN_POLE and STRUT_POLE.
    """
    import gc
    import torch

    request_id = re.sub(r"[^a-zA-Z0-9_-]", "", str(job_id or uuid.uuid4())) or str(uuid.uuid4())
    output_name = f"processed_video_{request_id}.webm"
    output_path = os.path.join(VIDEO_RESULTS_FOLDER, output_name)
    video_log = lambda message: log_video(f"[{worker_name}] {message}" if worker_name else message)

    video_log(f"[VIDEO:{request_id}] Received upload for video processing")
    set_video_progress(request_id, 1, "Upload received")
    if pole_model is None:
        pole_model = load_video_component_model()
    set_video_progress(request_id, 4, "Pole model ready")

    video_log(f"[VIDEO:{request_id}] Saved input video: {input_path}")
    set_video_progress(request_id, 8, "Input video saved")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        safe_remove_file(input_path, "video input")
        raise ValueError("Unable to read uploaded video")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        output_width, output_height = width, height
        if VIDEO_OUTPUT_MAX_WIDTH > 0 and width > VIDEO_OUTPUT_MAX_WIDTH:
            output_width = VIDEO_OUTPUT_MAX_WIDTH
            output_height = int(round(height * (output_width / max(1, width))))
        output_scale_x = output_width / max(1, width)
        output_scale_y = output_height / max(1, height)
        duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0

        trim_start = max(0.0, float(trim_start or 0.0))
        trim_duration = max(1.0, min(30.0, float(trim_duration or 30.0)))
        if duration > 0:
            trim_start = min(trim_start, max(0.0, duration - 0.1))
        trim_end = min(trim_start + trim_duration, duration) if duration > 0 else trim_start + trim_duration
        effective_trim_duration = max(0.0, trim_end - trim_start)
        start_frame = int(round(trim_start * fps))
        max_frames = int(max(1, round(effective_trim_duration * fps)))
        sample_fps = VIDEO_SAMPLE_FPS
        min_pole_appearances = max(2, int(MIN_POLE_VISIBLE_SECONDS * sample_fps))
        temporal_gap_max_frames = max(1, int(TEMPORAL_GAP_MAX_SECONDS * sample_fps))
        sample_step = fps / sample_fps
        sampled_frame_limit = int(max(1, math.floor(effective_trim_duration * sample_fps)))
        sample_indices = [
            int(round(start_frame + (i * sample_step)))
            for i in range(sampled_frame_limit)
        ]
        sample_index_set = set(sample_indices)
        video_log(
            f"[VIDEO:{request_id}] Metadata fps={fps:.2f}, frames={frame_count}, "
            f"size={width}x{height}, output={output_width}x{output_height}, "
            f"infer_imgsz={VIDEO_INFER_IMGSZ}, sample_fps={sample_fps:g}, duration={duration:.2f}s"
        )
        video_log(
            f"[VIDEO:{request_id}] Extracting {sample_fps:g} FPS from {trim_start:.2f}s to {trim_end:.2f}s "
            f"({sampled_frame_limit} deterministic sampled frames); output remains {fps:.2f} FPS for smooth playback"
        )
        video_log(
            f"[VIDEO:{request_id}] Thresholds: sample_fps={sample_fps:g}, "
            f"min_appearances={min_pole_appearances}, temporal_gap_max={temporal_gap_max_frames} frames"
        )
        set_video_progress(request_id, 12, "Metadata loaded and trim window prepared")
        tracker_backend = _video_tracker_backend()
        if tracker_backend == "bytetrack":
            video_log(f"[VIDEO:{request_id}] Using ByteTrack tracker")
            set_video_progress(request_id, 14, "Using ByteTrack tracker")
        else:
            video_log(f"[VIDEO:{request_id}] lap package missing; using built-in IoU/proximity tracker fallback")
            set_video_progress(request_id, 14, "Preparing video tracker")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # VP8/WebM is browser-playable; OpenCV mp4v MP4 often shows
        # "No video with supported format" in Chrome/Edge.
        fourcc = cv2.VideoWriter_fourcc(*"VP80")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
        if not writer.isOpened():
            raise ValueError("Unable to create processed video")

        tracks = {}
        class_counts = defaultdict(int)
        processed_frames = 0
        sampled_frames = 0
        last_progress_sample = 0
        last_frame_detections = []

        while processed_frames < max_frames:
            ok, frame = cap.read()
            if not ok:
                video_log(f"[VIDEO:{request_id}] Frame read stopped at frame {processed_frames}")
                break

            source_frame_index = start_frame + processed_frames
            frame_time = source_frame_index / fps if fps > 0 else trim_start
            should_sample = source_frame_index in sample_index_set
            annotated = (
                cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA)
                if output_width != width or output_height != height
                else frame.copy()
            )

            if should_sample:
                sampled_frames += 1
                sharpness = _frame_sharpness(frame)
                last_frame_detections = []
                if tracker_backend == "bytetrack":
                    results = pole_model.track(
                        frame,
                        persist=True,
                        tracker="bytetrack.yaml",
                        conf=0.25,
                        iou=0.45,
                        imgsz=VIDEO_INFER_IMGSZ,
                        verbose=False
                    )
                else:
                    results = pole_model(
                        frame,
                        conf=0.25,
                        iou=0.45,
                        imgsz=VIDEO_INFER_IMGSZ,
                        verbose=False
                    )

                for res in results:
                    if not getattr(res, "boxes", None):
                        continue
                    for box_obj in res.boxes:
                        cls_id = int(box_obj.cls)
                        raw_label = pole_model.names.get(cls_id, str(cls_id)) if hasattr(pole_model.names, "get") else pole_model.names[cls_id]
                        label = _normalise_video_pole_label(raw_label)
                        if label not in {"MAIN_POLE", "STRUT_POLE"}:
                            continue

                        conf = float(box_obj.conf)
                        bbox = [int(v) for v in box_obj.xyxy[0].cpu().numpy().tolist()]
                        x1, y1, x2, y2 = _clip_bbox(bbox, width, height)
                        area = max(1, (x2 - x1) * (y2 - y1))
                        norm_area = area / max(1, width * height)
                        track_id = None
                        if getattr(box_obj, "id", None) is not None:
                            track_id = int(box_obj.id.item())
                        if track_id is not None:
                            track_key = f"{label}:{track_id}"
                        else:
                            # Fallback mode: create short fragments, then merge them
                            # into physical poles after all sampled frames.
                            track_key = f"{label}:frame-{source_frame_index}:det-{len(tracks)}"

                        class_counts[label] += 1
                        current = tracks.get(track_key)
                        if current is None:
                            current = {
                                "label": label,
                                "track_id": track_id,
                                "appearances": 0,
                                "best_score": -1.0,
                                "best": None,
                                "observations": {},
                                "start_frame": source_frame_index,
                                "end_frame": source_frame_index,
                            }
                            tracks[track_key] = current

                        scored_bbox = [x1, y1, x2, y2]
                        best_score = _video_best_frame_score(conf, scored_bbox, width, height)
                        crop_bbox = _clip_bbox(bbox, width, height, padding=24)
                        _add_track_observation(current, source_frame_index, {
                            "confidence": conf,
                            "bbox": scored_bbox,
                            "crop_bbox": crop_bbox,
                            "frame_time": frame_time,
                            "frame_index": source_frame_index,
                            "sharpness": sharpness,
                            "area": area,
                            "score": best_score
                        })

                        last_frame_detections.append(([x1, y1, x2, y2], label, conf, track_id))

            for bbox, label, conf, track_id in last_frame_detections:
                _draw_video_detection(annotated, _scale_bbox(bbox, output_scale_x, output_scale_y), label, conf, track_id)

            writer.write(annotated)
            processed_frames += 1

            if sampled_frames and sampled_frames % 15 == 0 and sampled_frames != last_progress_sample:
                last_progress_sample = sampled_frames
                progress_pct = 15 + int((sampled_frames / max(1, sampled_frame_limit)) * 70)
                set_video_progress(
                    request_id,
                    progress_pct,
                    f"Sampled {sampled_frames}/{sampled_frame_limit} frames; raw fragments={len(tracks)}"
                )
                video_log(
                    f"[VIDEO:{request_id}] Sampled {sampled_frames}/{sampled_frame_limit} frames; "
                    f"pole_tracks={len(tracks)}"
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        video_log(f"[VIDEO:{request_id}] Consolidating {len(tracks)} tracker fragments into physical poles")
        set_video_progress(request_id, 88, "Consolidating track fragments into physical poles")
        fragment_groups = _merge_pole_track_fragments(tracks, width, temporal_gap_max_frames)
        physical_poles = _merge_temporal_pole_events(fragment_groups, width, temporal_gap_max_frames)
        persistent_poles = [pole for pole in physical_poles if int(pole.get("appearances", 0)) >= min_pole_appearances]
        removed = len(physical_poles) - len(persistent_poles)
        if removed > 0:
            video_log(
                f"[VIDEO:{request_id}] Filtered {removed} short-lived pole group(s) "
                f"below {min_pole_appearances} sampled-frame appearances"
            )
        physical_poles = persistent_poles
        video_log(
            f"[VIDEO:{request_id}] Temporal pole-event merge: "
            f"fragment_groups={len(fragment_groups)}, physical_poles={len(physical_poles)}"
        )
        for idx, pole in enumerate(physical_poles, start=1):
            best = pole["best"]
            video_log(
                f"[VIDEO:{request_id}] Physical pole {idx}: label={pole['label']} "
                f"fragments={pole['fragments']} frames={pole['appearances']} "
                f"track_ids={pole['track_ids']} time_range={pole.get('first_time', best['frame_time']):.2f}-"
                f"{pole.get('last_time', best['frame_time']):.2f}s best_time={best['frame_time']:.2f}s"
            )

        video_log(f"[VIDEO:{request_id}] Selecting one best frame per physical pole")
        set_video_progress(request_id, 92, "Selecting one best frame per physical pole")
        detections = []
        for idx, pole in enumerate(physical_poles, start=1):
            best = pole.get("best")
            if not best:
                continue

            detections.append({
                "label": pole["label"],
                "confidence": best["confidence"],
                "bbox": best["bbox"],
                "source": MODEL_PATHS["video_component"],
                "details": {
                    "type": "strut_pole" if pole["label"] == "STRUT_POLE" else "main_pole",
                    "track_id": pole["track_ids"][0] if pole.get("track_ids") else None,
                    "track_ids": [tid for tid in pole["track_ids"] if tid is not None],
                    "track_fragments": pole["fragments"],
                    "appearances": pole["appearances"],
                    "frame_time": round(best["frame_time"], 2),
                    "sharpness": round(best["sharpness"], 1),
                    "pole_area": int(best["area"]),
                    "best_frame_rank": idx,
                    "attributes": []
                }
            })

        detections = sorted(
            detections,
            key=lambda d: (d["label"], -float(d.get("confidence", 0)))
        )
        detected_classes = {
            label: sum(1 for d in detections if d["label"] == label)
            for label in ["MAIN_POLE", "STRUT_POLE"]
        }
        detected_classes = {k: v for k, v in detected_classes.items() if v > 0}
        processed_duration = processed_frames / fps if fps > 0 else (trim_end - trim_start)
        has_strut = any(d["label"] == "STRUT_POLE" for d in detections)
        survey_q = {
            "strut_pole": "Yes" if has_strut else "No",
            "strut_pole_count": sum(1 for d in detections if d["label"] == "STRUT_POLE"),
            "main_pole_count": sum(1 for d in detections if d["label"] == "MAIN_POLE"),
            "best_frame_count": len([d for d in detections if d["label"] in {"MAIN_POLE", "STRUT_POLE"}]),
            "raw_track_fragments": len(tracks),
            "physical_pole_count": len(physical_poles),
            "video_sampling_fps": sample_fps,
            "component_attributes": []
        }
        video_log(
            f"[VIDEO:{request_id}] Complete: sampled_frames={sampled_frames}, "
            f"raw_track_fragments={len(tracks)}, physical_poles={len(physical_poles)}, "
            f"detections={len(detections)}, class_counts={detected_classes}"
        )
        video_log(f"[VIDEO:{request_id}] Output video: {output_path}")
        set_video_progress(request_id, 100, "Video analysis complete", status="complete")

        return {
            "video_url": f"/static/results/{output_name}",
            "processed_video_url": f"/static/results/{output_name}",
            "detections": detections,
            "class_counts": detected_classes,
            "frame_detection_counts": dict(class_counts),
            "survey_questionnaire": survey_q,
            "master": {
                "final_class": "video_pole_detection",
                "voltage": "VIDEO",
                "pole_id": "Not Found",
                "ocr_text_lines": [],
                "reason": f"Sampled {sampled_frames} frames at {sample_fps:g} FPS, wrote smooth output at {fps:.2f} FPS, merged {len(tracks)} tracker fragments into {len(physical_poles)} physical pole(s), and selected one best frame per pole. Only MAIN_POLE and STRUT_POLE were processed.",
                "confidence": "high" if detections else "low",
                "pole_type": "video",
                "pole_status": "tracked",
                "model_summary": {
                    "pole_detector": MODEL_PATHS["video_component"],
                    "tracker": "ByteTrack" if tracker_backend == "bytetrack" else "IoU/proximity fallback",
                    "sampling_fps": sample_fps
                }
            },
            "width": width,
            "height": height,
            "duration": duration,
            "trim_start": trim_start,
            "trim_duration": round(processed_duration, 2),
            "processed_frames": sampled_frames,
            "pipeline": [
                "Video",
                f"Extract {sample_fps:g} FPS",
                "Pole Detector (YOLOv8)",
                "ByteTrack",
                "One Track = One Pole",
                "Best Frame Selection",
                "Survey Form Auto-fill"
            ],
        }
    finally:
        cap.release()
        if "writer" in locals():
            writer.release()
        safe_remove_file(input_path, "video input")
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

def process_video_file(file_stream, trim_start=0.0, trim_duration=30.0, job_id=None):
    """Compatibility wrapper for any direct callers; /predict_video now queues saved paths."""
    request_id = re.sub(r"[^a-zA-Z0-9_-]", "", str(job_id or uuid.uuid4())) or str(uuid.uuid4())
    input_path = os.path.join(UPLOADS_FOLDER, f"video_{request_id}.mp4")
    with open(input_path, "wb") as f:
        f.write(file_stream.read())
    return process_video_path(input_path, trim_start=trim_start, trim_duration=trim_duration, job_id=request_id)

start_video_workers(
    process_video_path,
    MODEL_PATHS["video_component"],
    log_func=log_video,
    max_workers=int(os.environ.get("MAX_VIDEO_WORKERS", "2") or 2),
)

@app.route('/predict_video', methods=['POST'])
@login_required
def predict_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    try:
        trim_start = float(request.form.get("trim_start", 0) or 0)
        trim_duration = float(request.form.get("trim_duration", 30) or 30)
        job_id = re.sub(r"[^a-zA-Z0-9_-]", "", str(request.form.get("job_id") or uuid.uuid4())) or str(uuid.uuid4())
        input_path = os.path.join(UPLOADS_FOLDER, f"video_{job_id}.mp4")
        with open(input_path, "wb") as f:
            f.write(request.files['video'].read())
        log_video(f"[VIDEO] /predict_video called trim_start={trim_start}, trim_duration={trim_duration}, job_id={job_id}")
        set_video_progress(job_id, 0, "Queued", status="queued")
        enqueue_video_job({
            "job_id": job_id,
            "input_path": input_path,
            "trim_start": trim_start,
            "trim_duration": trim_duration,
        })
        return jsonify({"job_id": job_id, "status": "queued"})
    except Exception as e:
        import traceback
        traceback.print_exc()
        if 'job_id' in locals():
            set_video_progress(job_id, 100, f"Video analysis failed: {str(e)}", status="failed")
        return jsonify({"error": f"Video Inference Error: {str(e)}"}), 500

@app.route('/api/video_progress/<job_id>')
@login_required
def video_progress(job_id):
    clean_job_id = re.sub(r"[^a-zA-Z0-9_-]", "", str(job_id))
    queued_job = get_video_job_status(clean_job_id, mark_fetched=True)
    progress = VIDEO_PROGRESS.get(clean_job_id, {})
    if not queued_job:
        percent = int(progress.get("percent", progress.get("progress", 0)) or 0)
        return jsonify({
            "status": progress.get("status", "waiting"),
            "progress": percent,
            "percent": percent,
            "message": progress.get("message", "Waiting for video job"),
            "queue_position": 0,
        })

    percent = int(progress.get("percent", queued_job.get("progress", 0)) or 0)
    response = {
        "status": queued_job.get("status", "queued"),
        "progress": percent,
        "percent": percent,
        "message": progress.get("message", queued_job.get("message", "")),
        "queue_position": queued_job.get("queue_position", 0),
        "worker": queued_job.get("worker"),
    }
    if queued_job.get("status") == "complete":
        response["result"] = queued_job.get("result") or {}
    if queued_job.get("status") == "failed":
        response["error"] = queued_job.get("error") or progress.get("message") or "Video processing failed"
    return jsonify(response)

# =========================
# API ENDPOINTS
# =========================
@app.route('/admin/logs')
@admin_required
def audit_logs():
    conn = get_db_connection()
    logs = conn.execute('SELECT * FROM activity_logs ORDER BY timestamp DESC LIMIT 100').fetchall()
    conn.close()
    return render_template('audit_logs.html', logs=logs)

@app.route('/api/save_asset', methods=['POST'])
@login_required
def save_asset():
    data = request.json # { images: [{b64, detections, pole_angle}], master: {final_class, voltage, reason} }
    asset_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    worker_name = session['user']
    
    master = data.get('master', {})
    
    conn = get_db_connection()
    try:
        # DEEP LOGGING: Header
        print(f"[DB_LOG] save_asset Header: ID={asset_id}, Worker={worker_name}, Class={master.get('final_class')}, Volt={master.get('voltage')}")

        # 1. Save Asset Header
        a_class = master.get('final_class')
        a_volt  = master.get('voltage')
        a_reason = master.get('reason')
        
        a_pole_id = master.get('pole_id', 'Not Found')
        
        if isinstance(a_class, (dict, list)): a_class = json.dumps(a_class)
        if isinstance(a_volt, (dict, list)):  a_volt = json.dumps(a_volt)
        if isinstance(a_reason, (dict, list)): a_reason = json.dumps(a_reason)

        conn.execute('''
            INSERT INTO assets (id, worker_name, status, timestamp, asset_class, voltage, reason, pole_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (str(asset_id), str(worker_name), 'pending', str(timestamp), 
              str(a_class) if a_class is not None else None,
              str(a_volt) if a_volt is not None else None,
              str(a_reason) if a_reason is not None else None,
              str(a_pole_id)))
        
        # 2. Save Images
        for idx, img_data in enumerate(data['images']):
            # Ensure detections is properly serialized
            dets = img_data['detections']
            det_str = json.dumps(dets) if not isinstance(dets, str) else dets
            
            # DEEP LOGGING: Image
            print(f"[DB_LOG] save_asset Image[{idx}]: b64_len={len(img_data['image_b64']) if img_data.get('image_b64') else 'NONE'}, dets_len={len(det_str)}")

            # Extraction and File Save
            raw_b64 = str(img_data.get('image_b64', ''))
            b64_core = raw_b64.split(',').pop().strip()
            
            # Generate unique filename
            img_filename = f"{uuid.uuid4()}.jpg"
            img_path = os.path.join(UPLOADS_FOLDER, img_filename)
            
            # Save to disk
            try:
                with open(img_path, "wb") as f:
                    f.write(base64.b64decode(b64_core))
            except Exception as e:
                print(f"[DISK_ERROR] Failed to save image: {e}")
                raise e

            conn.execute('''
                INSERT INTO asset_images (asset_id, image_b64, detections, pole_angle)
                VALUES (?, ?, ?, ?)
            ''', (str(asset_id), img_filename, det_str, float(img_data.get('pole_angle', 0.0))))
            
        conn.commit()
        log_activity(worker_name, "asset_submission", f"Asset: {asset_id}, Images: {len(data['images'])}")
    except Exception as e:
        conn.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        conn.close()
    
    return jsonify({"status": "success", "asset_id": asset_id})

@app.route('/api/save_draft', methods=['POST'])
@login_required
def save_draft():
    data = request.json # { id, type: 'worker'|'admin', data: any }
    draft_id = data.get('id')
    dtype = data.get('type', 'worker')
    content = data.get('data')
    
    # Force content to string if it's a dict/list
    if isinstance(content, (dict, list)):
        content = json.dumps(content)
    else:
        content = str(content) if content is not None else ""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    conn = get_db_connection()
    conn.execute('''
        INSERT OR REPLACE INTO drafts (id, type, data, timestamp)
        VALUES (?, ?, ?, ?)
    ''', (draft_id, dtype, content, timestamp))
    conn.commit()
    conn.close()
    return jsonify({"status": "success"})

@app.route('/api/get_draft/<draft_id>')
@login_required
def get_draft(draft_id):
    conn = get_db_connection()
    draft = conn.execute('SELECT * FROM drafts WHERE id = ?', (draft_id,)).fetchone()
    conn.close()
    if draft:
        return jsonify({"status": "success", "data": draft['data']})
    return jsonify({"status": "error", "message": "Draft not found"}), 404

@app.route('/api/download_annotated/<asset_id>')
@login_required
def download_annotated(asset_id):
    """Generates and serves the annotated image for an asset."""
    conn = get_db_connection()
    row = conn.execute('SELECT image_b64, detections FROM asset_images WHERE asset_id = ? LIMIT 1', (asset_id,)).fetchone()
    conn.close()
    
    if not row:
        return "Asset image not found", 404
        
    from report_generator import annotate_image
    annotated_b64 = annotate_image(row['image_b64'], parse_db_json(row['detections']))
    
    # Ensure any prefix is stripped before final decode
    try:
        cleaned = clean_b64(annotated_b64)
        img_data = base64.b64decode(cleaned)
        buffer = io.BytesIO(img_data)
        
        return send_file(
            buffer,
            mimetype='image/jpeg',
            as_attachment=True,
            download_name=f"Annotated_Asset_{asset_id[:8]}.jpg"
        )
    except Exception as e:
        print("Base64 decode failed:", e)
        return jsonify({"error": "Invalid image data"}), 400
@app.route('/api/get_assets')
@login_required
def get_assets():
    status = request.args.get('status')
    
    conn = get_db_connection()
    # Get assets along with the first image as a thumbnail
    query = '''
        SELECT a.*, i.image_b64 as thumbnail, i.detections as detections
        FROM assets a
        LEFT JOIN asset_images i ON i.id = (
            SELECT id FROM asset_images WHERE asset_id = a.id LIMIT 1
        )
        WHERE 1=1
    '''
    params = []
    
    if status:
        query += ' AND a.status = ?'
        params.append(status)
    
    if session['role'] != 'admin':
        query += ' AND a.worker_name = ?'
        params.append(session['user'])
    
    query += ' ORDER BY a.timestamp DESC'
    rows = conn.execute(query, params).fetchall()
    conn.close()
    
    # Clean up results
    data = []
    for r in rows:
        d = dict(r)
        if d.get('detections'):
            d['detections'] = parse_db_json(d['detections'])
        data.append(d)
    
    return jsonify(data)

@app.route('/admin/asset/pdf/<asset_id>')
@admin_required
def export_asset_pdf(asset_id):
    generate_asset_pdf, _, _, _ = load_report_generators()
    conn = get_db_connection()
    asset_row = conn.execute('SELECT * FROM assets WHERE id = ?', (asset_id,)).fetchone()
    if not asset_row:
        conn.close()
        return "Asset not found", 404
        
    image_rows = conn.execute('SELECT * FROM asset_images WHERE asset_id = ?', (asset_id,)).fetchall()
    conn.close()
    
    asset_data = dict(asset_row)
    asset_data['images'] = [dict(r) for r in image_rows]
    for img in asset_data['images']:
        img['detections'] = parse_db_json(img['detections'])

    pdf_buffer = generate_asset_pdf(asset_data)
    filename = f"Inspection_Report_{asset_id[:8]}.pdf"
    
    return send_file(pdf_buffer, download_name=filename, as_attachment=True, mimetype='application/pdf')

@app.route('/admin/asset/excel/<asset_id>')
@admin_required
def export_asset_excel(asset_id):
    _, generate_asset_excel, _, _ = load_report_generators()
    conn = get_db_connection()
    asset_row = conn.execute('SELECT * FROM assets WHERE id = ?', (asset_id,)).fetchone()
    if not asset_row:
        conn.close()
        return "Asset not found", 404
        
    image_rows = conn.execute('SELECT * FROM asset_images WHERE asset_id = ?', (asset_id,)).fetchall()
    conn.close()
    
    asset_data = dict(asset_row)
    asset_data['images'] = [dict(r) for r in image_rows]
    for img in asset_data['images']:
        img['detections'] = parse_db_json(img['detections'])

    excel_buffer = generate_asset_excel(asset_data)
    filename = f"Detection_Log_{asset_id[:8]}.xlsx"
    
    return send_file(excel_buffer, download_name=filename, as_attachment=True, 
                     mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

@app.route('/api/update_asset_status', methods=['POST'])
@login_required
def update_asset_status():
    data = request.json
    asset_id = data.get('asset_id')
    status = data.get('status')
    
    conn = get_db_connection()
    conn.execute('UPDATE assets SET status = ? WHERE id = ?', (status, asset_id))
    conn.commit()
    conn.close()
    
    log_activity(session['user'], "asset_status_update", f"Asset: {asset_id}, Status: {status}")
    
    # When admin APPROVES → auto-export annotations as YOLO training data
    if status == 'approved':
        try:
            export_asset_to_training, _ = load_training_pipeline()
            result = export_asset_to_training(asset_id, approved_by=session['user'])
            log_activity(
                session['user'],
                "training_data_exported",
                f"Asset: {asset_id} | Exported: {result['exported']} images | Classes: {json.dumps(result['classes'])} | Pool: {result['total_pool']}"
            )
        except Exception as e:
            print(f"[app] Training export error: {e}")
    
    return jsonify({"status": "success"})

@app.route('/api/delete_asset/<asset_id>', methods=['DELETE'])
@admin_required
def delete_asset(asset_id):
    conn = get_db_connection()
    try:
        # Note: Foreign key cascade should handle asset_images if configured, 
        # but let's be explicit just in case.
        conn.execute('DELETE FROM asset_images WHERE asset_id = ?', (asset_id,))
        conn.execute('DELETE FROM assets WHERE id = ?', (asset_id,))
        conn.commit()
        log_activity(session['user'], "asset_delete_full", f"Permanently deleted Asset: {asset_id}")
        return jsonify({"status": "success"})
    except Exception as e:
        conn.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/update_asset_detections', methods=['POST'])
@admin_required
def update_asset_detections():
    data = request.json
    asset_id = data.get('asset_id')
    image_updates = data.get('updates') # List of {image_id, detections}
    
    conn = get_db_connection()
    try:
        for update in image_updates:
            # We need the relative index or ID. 
            # In asset_detail.html, we use the list order. 
            # Better to pass the image index or original ID if we had it.
            # Assuming updates are {image_index: int, detections: list}
            
            # Re-fetch images to match index correctly
            image_rows = conn.execute('SELECT id FROM asset_images WHERE asset_id = ? ORDER BY id ASC', (asset_id,)).fetchall()
            if update['index'] < len(image_rows):
                img_db_id = image_rows[update['index']]['id']
                conn.execute('UPDATE asset_images SET detections = ? WHERE id = ?', 
                             (json.dumps(update['detections']), img_db_id))
        
        conn.commit()
        log_activity(session['user'], "asset_annotation_edit", f"Modified annotations for Asset: {asset_id}")
        return jsonify({"status": "success"})
    except Exception as e:
        conn.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/delete_asset_image', methods=['POST'])
@admin_required
def delete_asset_image():
    data = request.json
    image_id = data.get('image_id')
    
    conn = get_db_connection()
    try:
        conn.execute('DELETE FROM asset_images WHERE id = ?', (image_id,))
        conn.commit()
        log_activity(session['user'], "asset_image_delete", f"Deleted Image ID: {image_id}")
        return jsonify({"status": "success"})
    except Exception as e:
        conn.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        conn.close()



@app.route('/api/get_asset_history/<asset_id>')
@admin_required
def get_asset_history(asset_id):
    conn = get_db_connection()
    logs = conn.execute('''
        SELECT user_name, action, details, timestamp
        FROM activity_logs 
        WHERE details LIKE ? 
        ORDER BY timestamp ASC
    ''', (f'%{asset_id}%',)).fetchall()
    conn.close()
    return jsonify([dict(l) for l in logs])

# =========================
# TRAINING PIPELINE API
# =========================
@app.route('/api/training_stats')
@login_required
def training_stats():
    """Returns training pool stats for authenticated dashboards."""
    try:
        _, get_training_stats = load_training_pipeline()
        stats = get_training_stats()
        return jsonify(build_training_dashboard_stats(stats))
    except Exception as e:
        print(f"[app] Training stats unavailable: {e}")
        return jsonify({
            "total_samples": 0,
            "total_classes": 0,
            "total_annotations": 0,
            "avg_confidence": 0,
            "by_class": {},
            "class_confidence": {},
            "models": [],
            "datasets": [],
            "status": "unavailable"
        })

@app.route('/api/training_export/<asset_id>', methods=['POST'])
@admin_required
def manual_training_export(asset_id):
    """Manually trigger export for a specific asset (re-export if needed)."""
    try:
        export_asset_to_training, _ = load_training_pipeline()
        result = export_asset_to_training(asset_id, approved_by=session['user'])
        return jsonify({"status": "success", **result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)
