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
from datetime import datetime
import sqlite3
import json
import os
from functools import wraps
from concurrent.futures import ThreadPoolExecutor

from config import DB_TYPE, DB_NAME, PG_HOST, PG_PORT, PG_USER, PG_PASS, PG_DB

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
MODEL_PATHS = {
    "pole": "backup_models/best (2).pt",
    "components": "backup_models/channel_12class_v2.pt",
    "insulator": "backup_models/insulator_model.pt",
    "shed": "models/shed_model.pt",
    "conductor_unet": "models/conductor_unet.pth",
}

def load_detection_models(load_unet=False):
    """Load heavy AI dependencies only when image prediction is requested."""
    global pipeline_engine, unet_model, device
    if pipeline_engine is not None and (not load_unet or unet_model is not None):
        return

    import torch
    from pipeline import InfrastructurePipeline

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
    temp_filename = f"temp_ocr_{uuid.uuid4()}.jpg"
    try:
        with open(temp_filename, "wb") as f:
            f.write(blob)
        return read_pole_id_with_rapidocr(temp_filename)
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

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
    temp_filename = f"temp_{uuid.uuid4()}.jpg"
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
            if float(conf) < 0.40:
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
            if float(conf) < 0.40:
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

        # Cleanup temporary image file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
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
        return jsonify(stats)
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
