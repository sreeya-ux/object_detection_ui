from flask import Flask, render_template, request, jsonify, session, redirect, url_for, make_response, send_file, send_from_directory
from werkzeug.security import check_password_hash, generate_password_hash
import requests
import csv
import io
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import uuid
from datetime import datetime
import sqlite3
import json
import os
from functools import wraps

import torch
import segmentation_models_pytorch as smp
from pipeline import InfrastructurePipeline
from training_pipeline import export_asset_to_training, get_training_stats
from report_generator import generate_asset_pdf, generate_asset_excel, generate_global_excel, generate_global_pdf
import psycopg2
from psycopg2.extras import RealDictCursor
from config import DB_TYPE, DB_NAME, PG_HOST, PG_PORT, PG_USER, PG_PASS, PG_DB
from flask_cors import CORS

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

# Master Rule-Engine Pipeline
# Centralizes component detection, classification, and rule-based logic.
pipeline_engine = InfrastructurePipeline(
    component_model_path="models/pole_model.pt",
    insulator_model_path="models/insulator_new.pt",
    shed_model_path="models/shed_model.pt",
    crossarm_model_path="models/best_components.pt"
)

# Load YOLOv8-seg model specifically for Conductor (Cable) Instance Segmentation
cable_model = YOLO("models/cable_best.pt")


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
    if DB_TYPE == "postgres":
        try:
            conn = psycopg2.connect(
                host=PG_HOST, port=PG_PORT, database=PG_DB,
                user=PG_USER, password=PG_PASS
            )
            return DBConn(conn, is_pg=True)
        except Exception as e:
            print(f"PostgreSQL Error: {e}")
            raise e
    else:
        conn = sqlite3.connect(DB_NAME)
        conn.row_factory = sqlite3.Row
        return DBConn(conn, is_pg=False)

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
    # Dynamic check for CUSTOM_DOMAIN to display on the enterprise login screen
    try:
        from config import CUSTOM_DOMAIN
        if CUSTOM_DOMAIN:
            return f"https://{CUSTOM_DOMAIN}"
    except ImportError:
        pass
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
        username = request.form.get('username').strip()
        password = request.form.get('password').strip()
        
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()
        
        if user:
            if check_password_hash(user['password'], password):
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
def merge_collinear_conductors(conductors, img_w, img_h):
    """
    Groups and merges collinear conductor segments.
    Each conductor is represented by a dict containing at least 'bbox' and 'polygon'.
    """
    if not conductors:
        return []
    
    import math
    
    def get_endpoints(poly):
        if not poly or len(poly) < 2:
            return None, None
        max_dist = -1
        p1, p2 = None, None
        pts = np.array(poly)
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = np.sum((pts[i] - pts[j])**2)
                if d > max_dist:
                    max_dist = d
                    p1, p2 = pts[i], pts[j]
        if p1 is not None:
            return tuple(p1), tuple(p2)
        return tuple(poly[0]), tuple(poly[-1])
        
    def point_to_line_dist(pt, lp1, lp2):
        x0, y0 = pt
        x1, y1 = lp1
        x2, y2 = lp2
        dx = x2 - x1
        dy = y2 - y1
        denom = math.sqrt(dx*dx + dy*dy)
        if denom == 0:
            return math.sqrt((x0-x1)**2 + (y0-y1)**2)
        return abs(dy*x0 - dx*y0 + x2*y1 - y2*x1) / denom

    def should_merge(d1, d2):
        poly1 = d1.get("polygon")
        poly2 = d2.get("polygon")
        if not poly1 or not poly2:
            return False
            
        e11, e12 = get_endpoints(poly1)
        e21, e22 = get_endpoints(poly2)
        if not e11 or not e21:
            return False
            
        v1 = np.array(e12) - np.array(e11)
        v2 = np.array(e22) - np.array(e21)
        
        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)
        if len1 == 0 or len2 == 0:
            return False
            
        u1 = v1 / len1
        u2 = v2 / len2
        
        # 1. Nearly parallel check: cosine of angle difference
        cos_theta = abs(np.dot(u1, u2))
        if cos_theta < 0.95:  # ~18 degrees
            return False
            
        # 2. Collinearity check: distance of segment endpoints to each other's lines
        d_c = point_to_line_dist(e21, e11, e12)
        d_d = point_to_line_dist(e22, e11, e12)
        d_a = point_to_line_dist(e11, e21, e22)
        d_b = point_to_line_dist(e12, e21, e22)
        
        if max(d_c, d_d, d_a, d_b) > 30.0:  # 30 pixels threshold
            return False
            
        # 3. Gap check
        gap = min(
            np.linalg.norm(np.array(e11) - np.array(e21)),
            np.linalg.norm(np.array(e11) - np.array(e22)),
            np.linalg.norm(np.array(e12) - np.array(e21)),
            np.linalg.norm(np.array(e12) - np.array(e22))
        )
        max_gap = max(400.0, img_w * 0.35)
        if gap > max_gap:
            return False
            
        return True

    def merge_two(d1, d2):
        poly1 = d1.get("polygon", [])
        poly2 = d2.get("polygon", [])
        combined_pts = np.array(poly1 + poly2, dtype=np.int32)
        hull = cv2.convexHull(combined_pts)
        merged_poly = [[int(pt[0][0]), int(pt[0][1])] for pt in hull]
        
        b1 = d1.get("bbox", [0,0,0,0])
        b2 = d2.get("bbox", [0,0,0,0])
        merged_bbox = [
            min(b1[0], b2[0]),
            min(b1[1], b2[1]),
            max(b1[2], b2[2]),
            max(b1[3], b2[3])
        ]
        
        avg_thick = (d1.get("thickness", 0.0) + d2.get("thickness", 0.0)) / 2.0
        max_conf = max(d1.get("confidence", 0.90), d2.get("confidence", 0.90))
        
        return {
            "label": "conductor",
            "confidence": max_conf,
            "bbox": merged_bbox,
            "polygon": merged_poly,
            "source": "models/cable_best.pt",
            "thickness": round(avg_thick, 1)
        }

    current = list(conductors)
    merged_any = True
    while merged_any:
        merged_any = False
        i = 0
        while i < len(current):
            j = i + 1
            while j < len(current):
                if should_merge(current[i], current[j]):
                    new_det = merge_two(current[i], current[j])
                    current[i] = new_det
                    current.pop(j)
                    merged_any = True
                    break
                else:
                    j += 1
            if merged_any:
                break
            i += 1
            
    return current

def process_image_file(file_stream):
    """
    Main diagnostic entry point.
    Combines Rule Engine (InfrastructurePipeline) with UNet Conductor Segmentation.
    """
    # Create a temporary file to run the pipeline.predict (which expects a path)
    import gc
    import psutil
    
    def log_mem(step):
        m = psutil.Process().memory_info().rss / (1024 * 1024)
        print(f"[Memory] {step}: {m:.1f} MB")

    log_mem("Start Inference")
    temp_filename = f"temp_{uuid.uuid4()}.jpg"
    try:
        with open(temp_filename, "wb") as f:
            f.write(file_stream.read())
        
        # 1. Run the Rule Engine Pipeline (Optimized to single scale in pipeline.py)
        log_mem("Before Pipeline")
        pipe_res = pipeline_engine.predict(temp_filename, visualize=False)
        log_mem("After Pipeline")
        gc.collect()

        # Reload image for UNet processing and base64 response
        img = cv2.imread(temp_filename)
        h, w = img.shape[:2]
        
        # 2. Process Conductors with YOLO Segmentation Model (cable_best.pt)
        # imgsz=800: stable memory footprint; conf=0.10 catches low-confidence wires
        mask_resized = np.zeros((h, w), dtype=np.uint8)
        try:
            cable_results = cable_model(temp_filename, imgsz=800, conf=0.10, verbose=False)
            if cable_results and len(cable_results) > 0:
                masks_obj = cable_results[0].masks
                if masks_obj is not None and masks_obj.xy is not None:
                    for poly_pts in masks_obj.xy:
                        if len(poly_pts) > 0:
                            cv2.fillPoly(mask_resized, [poly_pts.astype(np.int32)], 255)
            log_mem("After Cable YOLO")
        except Exception as e:
            print(f"[Cable Inference Error] {e}")

        # Explicitly free memory
        if 'cable_results' in locals():
            del cable_results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()



        # Thickness Measurement via Distance Transform & Skeletonize
        dist = cv2.distanceTransform(mask_resized, cv2.DIST_L2, 5)
        from skimage.morphology import skeletonize
        skel = (skeletonize(mask_resized / 255.0) > 0).astype(np.uint8)

        # Bridge gaps for continuous polygons
        # Step 1: Dilate slightly to amplify hairline / faint mask pixels
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_dilated = cv2.dilate(mask_resized, dilate_kernel, iterations=1)
        # Step 2: Close with a larger kernel to bridge micro-gaps in thin wires
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_closed = cv2.morphologyEx(mask_dilated, cv2.MORPH_CLOSE, close_kernel)
        
        # 3. Create Hardware Blackout Mask (Wire Detection Last)
        # Prevents wires from "hallucinating" over insulators/poles
        hardware_mask = np.zeros((h, w), dtype=np.uint8)
        
        final_detections = []

        # A. Map Rule Engine Components to UI format (with OBB Polygons)
        # Each component in pipe_res is now (box, conf, angle, polygon)
        for ins in pipe_res.insulators:
            # Map to Hardware Mask with 5px buffer
            x1, y1, x2, y2 = [int(v) for v in ins.box]
            cv2.rectangle(hardware_mask, (max(0, x1-5), max(0, y1-5)), (min(w, x2+5), min(h, y2+5)), 255, -1)
            
            # Map to specific insulator subclass label
            ins_label = "INSULATOR"
            if ins.type_heuristic == "HT_disc" or ins.type_final == "disc":
                ins_label = "HT_DISC"
            elif ins.type_heuristic == "HT_pin":
                ins_label = "HT_PIN"
            elif ins.type_heuristic == "LT_pin":
                ins_label = "LT_PIN"
            elif ins.type_heuristic == "shackle_insulator" or ins.type_final == "shackle":
                ins_label = "SHACKLE_INSULATOR"
            elif ins.type_final == "pin":
                ins_label = "HT_PIN"
                
            final_detections.append({
                "label": ins_label,
                "confidence": float(ins.detection_conf),
                "bbox": [int(x) for x in ins.box],
                "polygon": ins.obb_polygon if hasattr(ins, 'obb_polygon') else None,
                "source": "models/insulator_new.pt",
                "details": {
                    "voltage": ins.voltage,
                    "shed_count": int(ins.shed_count),
                    "sheds": int(ins.shed_count),
                    "type": ins.type_final
                }
            })
        
        for ca in pipe_res.crossarms:
            # Map to Hardware Mask with 5px buffer
            x1, y1, x2, y2 = [int(v) for v in ca.box]
            cv2.rectangle(hardware_mask, (max(0, x1-5), max(0, y1-5)), (min(w, x2+5), min(h, y2+5)), 255, -1)
            
            # Map crossarm shape to standard UI category
            shape_lower = ca.shape.lower()
            if "v_arm" in shape_lower or "v_cross" in shape_lower or "v cross" in shape_lower:
                ui_label = "v_cross"
            elif "t_raising" in shape_lower or "t_rising" in shape_lower or "t-rising" in shape_lower or "t_arm" in shape_lower:
                ui_label = "t_rising"
            elif "tapping arm" in shape_lower or "tapping_channel" in shape_lower or "tapping channel" in shape_lower:
                ui_label = "tapping_channel"
            elif "side arm" in shape_lower or "side_arm_channel" in shape_lower or "side arm channel" in shape_lower:
                ui_label = "side_arm_channel"
            else:
                ui_label = "crossarm"

            final_detections.append({
                "label": ui_label,
                "confidence": float(ca.detection_conf) * 0.75,
                "bbox": [int(x) for x in ca.box],
                "polygon": ca.obb_polygon if hasattr(ca, 'obb_polygon') else None,
                "source": "models/best_components.pt",
                "details": {
                    "shape": ca.shape
                }
            })
        
        for po in pipe_res.all_poles:
            # Map to Hardware Mask with 5px buffer
            x1, y1, x2, y2 = [int(v) for v in po.box]
            # Avoid cutting continuous horizontal cables in half by not blacking out poles
            # cv2.rectangle(hardware_mask, (max(0, x1-5), max(0, y1-5)), (min(w, x2+5), min(h, y2+5)), 255, -1)
            
            final_detections.append({
                "label": "strut_pole" if po.pole_type == "strut_pole" else "pole",
                "confidence": float(po.detection_conf),
                "bbox": [int(x) for x in po.box],
                "polygon": po.obb_polygon if hasattr(po, 'obb_polygon') else None,
                "source": "models/pole_model.pt",
                "details": {
                    "type": po.pole_type,
                    "lean": round(float(po.lean_angle_deg), 1)
                }
            })

        for box, conf, poly in pipe_res.street_lights:
            # Map to Hardware Mask (Street lights are hardware too, wires shouldn't pass THROUGH them)
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(hardware_mask, (max(0, x1-5), max(0, y1-5)), (min(w, x2+5), min(h, y2+5)), 255, -1)
            
            final_detections.append({
                "label": "street_light",
                "confidence": float(conf) * 0.75,
                "bbox": [int(x) for x in box],
                "polygon": poly,
                "source": "models/best_components.pt",
                "details": {"type": "Standard Lamp"}
            })

        for label, box, conf, poly in pipe_res.others:
            # We add large 'other' items to the exclusion mask to prevent wire ghosts
            bw, bh = box[2]-box[0], box[3]-box[1]
            if bw > 100 or bh > 100:
                 cv2.rectangle(hardware_mask, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), 255, -1)

            final_detections.append({
                "label": label.lower().replace(" ", "_"),
                "confidence": float(conf),
                "bbox": [int(x) for x in box],
                "polygon": poly,
                "details": {"source": "AI Inference"}
            })

        # Collect main pole boxes (non-strut poles)
        main_pole_boxes = []
        for po in pipe_res.all_poles:
            if po.pole_type != "strut_pole":
                main_pole_boxes.append([int(v) for v in po.box])

        # --- Wire Discovery Phase 2: Exclude static hardware ---
        # Any wire detected INSIDE a hardware box is disqualified to reduce noise
        mask_final = cv2.bitwise_and(mask_closed, cv2.bitwise_not(hardware_mask))
        
        # B. Generate Conductor Polygons from clean mask
        contours, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw_conductors = []
        for c in contours:
            cx, cy, cw, ch = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            
            # --- 1. Basic Size Filter ---
            # Lowered to 60/60 to catch faint low-confidence wire fragments
            if cw + ch < 60 or area < 60: 
                continue
            
            # --- 2. Geometric "Clump" Filter ---
            # Real wires are elongated. Huge square clumps are usually noise or shadows.
            aspect_ratio = max(cw, ch) / max(1, min(cw, ch))
            solidity = area / (cw * ch)
            
            # If it's a large clumpy rectangle (high solidity, low elongation), it's likely noise
            if cw > 150 and ch > 150 and solidity > 0.5 and aspect_ratio < 1.8:
                continue

            # --- 2.5 Main Pole Association Filter (Top of Pole only) ---
            # Conductor must horizontally overlap with at least one detected main pole,
            # and it must be located near the top of that pole (excluding the lower body)
            overlaps_main_pole = False
            for p_box in main_pole_boxes:
                px1, py1, px2, py2 = p_box
                ph = py2 - py1
                # Relaxed to 0.45 to catch wires attached slightly lower on the pole
                threshold_y = py1 + max(150, int(ph * 0.45))
                
                # Check horizontal overlap AND vertical top-of-pole constraint
                if cx <= px2 and (cx + cw) >= px1 and cy <= threshold_y:
                    overlaps_main_pole = True
                    break
            if not overlaps_main_pole:
                continue
            
            # --- 3. Process Valid Wire ---
            # Use small epsilon (1.0 pixel) to preserve high-precision segment polygon shape
            # and prevent it from being simplified to a 2-point straight line
            epsilon = 1.0
            approx = cv2.approxPolyDP(c, epsilon, True)
            polygon = [[int(pt[0][0]), int(pt[0][1])] for pt in approx]
            
            # Localized thickness
            c_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(c_mask, [c], -1, 255, -1)
            local_skel = skel & (c_mask > 0)
            local_thickness = dist[local_skel > 0] * 2
            avg_thick = float(np.mean(local_thickness)) if len(local_thickness) > 0 else 0.0
            
            # Reject if the detected "wire" is physically impossible (too thick)
            # Raised to 100 so borderline thin segments aren't discarded
            if avg_thick > 100:
                continue
            
            raw_conductors.append({
                "label": "conductor",
                "confidence": 0.70,
                "bbox": [cx, cy, cx+cw, cy+ch],
                "polygon": polygon, # High precision YOLO segment polygon
                "source": "models/cable_best.pt",
                "thickness": round(avg_thick, 1)
            })

        # Merge collinear segments
        merged_conductors = merge_collinear_conductors(raw_conductors, w, h)
        
        # Calculate primary pole horizontal center
        p_cx = w // 2
        if main_pole_boxes:
            px1, py1, px2, py2 = main_pole_boxes[0]
            p_cx = (px1 + px2) // 2

        # Add all merged conductors to final detections
        for mc in merged_conductors:
            final_detections.append(mc)

        # Prepare Master Data (Asset Identity)
        master_data = {
            "final_class": pipe_res.final_class,
            "voltage": pipe_res.voltage,
            "pole_id": pipe_res.pole_id,
            "reason": pipe_res.reason,
            "confidence": pipe_res.confidence,
            "pole_lean_angle": pipe_res.pole_orientation.lean_angle_deg if pipe_res.pole_orientation else 0.0,
            "pole_type": pipe_res.pole_orientation.pole_type if pipe_res.pole_orientation else "none",
            "pole_status": pipe_res.pole_orientation.fault_severity if pipe_res.pole_orientation else "none",
            "model_summary": {
                "structural": "models/pole_model.pt",
                "insulator": "models/insulator_new.pt",
                "segmentation": "models/cable_best.pt"
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
        _, buffer = cv2.imencode('.jpg', img)
        img_b64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "detections": final_detections,
            "master": master_data,
            "survey_questionnaire": survey_q,
            "annotated_image": img_b64,
            "width": w,
            "height": h
        }
    finally:
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
        result = process_image_file(file_stream)
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
        parsed_img['detections'] = json.loads(img['detections'])
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
        img_dict['detections'] = json.loads(img_dict['detections'])
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
            # Single image: standard processing
            result = process_image_file(files[0])
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
    for idx, stream in enumerate(file_streams):
        try:
            print(f"DEBUG: Processing image {idx+1}/{len(file_streams)} in merged mode...")
            res = process_image_file(stream)
            if res:
                results[idx] = res
        except Exception as e:
            print(f"[ERROR] Failed to process image {idx+1} in merged mode: {e}")
            import traceback
            traceback.print_exc()

    valid_results = [r for r in results if r is not None]
    if not valid_results:
        return {"error": "No valid results generated from uploaded images"}

    # --- MERGE LOGIC ---
    # 1. Best Pole ID (OCR)
    best_pole_id = "Not Found"
    for r in valid_results:
        if r.get('master', {}).get('pole_id') and r['master']['pole_id'] != "Not Found":
            # Prefer IDs that aren't 'Not Found'
            best_pole_id = r['master']['pole_id']
            break

    # 2. Choose the 'Best' result as the Master
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

    # Inject the best OCR found across all images
    merged_result['master']['pole_id'] = best_pole_id
    
    # Add a flag that this was a merged result
    merged_result['master']['reason'] = f"[Merged from {len(valid_results)} images] " + merged_result['master']['reason']
    
    # Store all images in the result so the UI can display them maintaining correct index mapping
    merged_result['all_images'] = [r['annotated_image'] if r else None for r in results]
    merged_result['all_detections'] = [r['detections'] if r else [] for r in results]
    merged_result['all_dims'] = [{"width": r['width'], "height": r['height']} if r else {"width": 0, "height": 0} for r in results]
    merged_result['all_pole_ids'] = [r['master'].get('pole_id', 'Not Found') if r else 'Not Found' for r in results]

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
    annotated_b64 = annotate_image(row['image_b64'], json.loads(row['detections'] or '[]'))
    
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
            try:
                d['detections'] = json.loads(d['detections'])
            except:
                d['detections'] = []
        data.append(d)
    
    return jsonify(data)

@app.route('/admin/asset/pdf/<asset_id>')
@admin_required
def export_asset_pdf(asset_id):
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
        img['detections'] = json.loads(img['detections'])

    pdf_buffer = generate_asset_pdf(asset_data)
    filename = f"Inspection_Report_{asset_id[:8]}.pdf"
    
    return send_file(pdf_buffer, download_name=filename, as_attachment=True, mimetype='application/pdf')

@app.route('/admin/asset/excel/<asset_id>')
@admin_required
def export_asset_excel(asset_id):
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
        img['detections'] = json.loads(img['detections'])

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
@admin_required
def training_stats():
    """Returns training pool stats for the Admin dashboard."""
    stats = get_training_stats()
    return jsonify(stats)

@app.route('/api/training_export/<asset_id>', methods=['POST'])
@admin_required
def manual_training_export(asset_id):
    """Manually trigger export for a specific asset (re-export if needed)."""
    try:
        result = export_asset_to_training(asset_id, approved_by=session['user'])
        return jsonify({"status": "success", **result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
