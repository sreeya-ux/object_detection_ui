from flask import Flask, render_template, request, jsonify, session, redirect, url_for, make_response, send_file
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

# =========================
# GLOBAL INITIALIZATION
# =========================
app = Flask(__name__)
app.secret_key = "secret_key_for_session" # In production, use a strong random key
DB_PATH = 'database.db'

# Master Rule-Engine Pipeline
# Centralizes component detection, classification, and rule-based logic.
pipeline_engine = InfrastructurePipeline(
    component_model_path="dry_backup/best_whole.pt",
    insulator_model_path="dry_backup/best_insulator.pt",
    shed_model_path="dry_backup/best_disc.pt"
)

# Load UNet specifically for Conductor Instance Segmentation (ResNet34)
# Configured for BGR 512x512 input as per original training requirements.
unet_model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
unet_model.load_state_dict(torch.load("best_cable_unet.pth", map_location="cpu"))
unet_model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
unet_model.to(device)

# =========================
# DATABASE HELPERS
# =========================
def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

def clean_b64(b64_str):
    """Robustly strips prefixes and fixes padding for b64 strings."""
    if not b64_str: return ""
    if ',' in b64_str:
        b64_str = b64_str.split(',')[1]
    # Remove whitespace
    b64_str = b64_str.strip()
    # Add padding if needed
    missing_padding = len(b64_str) % 4
    if missing_padding:
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
    try:
        # Increased timeout to 2.0s to prevent false negatives under high load
        response = requests.get('http://127.0.0.1:4040/api/tunnels', timeout=2.0)
        if response.status_code == 200:
            tunnels = response.json().get('tunnels', [])
            for tunnel in tunnels:
                if tunnel.get('proto') == 'https':
                    return tunnel.get('public_url')
    except Exception as e:
        print(f"[Ngrok] Connection failed: {e}")
        return None
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
        
        # 2. Process Conductors with UNet Segmentation Model (ResNet34)
        input_img = cv2.resize(img, (512, 512)).transpose(2, 0, 1) / 255.0
        tensor = torch.tensor(input_img[None, ...], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            out = unet_model(tensor)
            mask = torch.sigmoid(out).squeeze().cpu().numpy()
            log_mem("After UNet")
            del out
        
        # Explicitly free memory
        del tensor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        mask_binary = (mask > 0.25).astype(np.uint8) * 255
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
            # Map to Hardware Mask with 5px buffer
            x1, y1, x2, y2 = [int(v) for v in ins.box]
            cv2.rectangle(hardware_mask, (max(0, x1-5), max(0, y1-5)), (min(w, x2+5), min(h, y2+5)), 255, -1)
            
            final_detections.append({
                "label": "insulator",
                "confidence": float(ins.detection_conf),
                "bbox": [int(x) for x in ins.box],
                "polygon": ins.obb_polygon if hasattr(ins, 'obb_polygon') else None,
                "details": {
                    "voltage": ins.voltage,
                    "shed_count": int(ins.shed_count),
                    "type": ins.type_final
                }
            })
        
        for ca in pipe_res.crossarms:
            # Map to Hardware Mask with 5px buffer
            x1, y1, x2, y2 = [int(v) for v in ca.box]
            cv2.rectangle(hardware_mask, (max(0, x1-5), max(0, y1-5)), (min(w, x2+5), min(h, y2+5)), 255, -1)
            
            final_detections.append({
                "label": "crossarm",
                "confidence": float(ca.detection_conf),
                "bbox": [int(x) for x in ca.box],
                "polygon": ca.obb_polygon if hasattr(ca, 'obb_polygon') else None,
                "details": {
                    "shape": ca.shape
                }
            })
        
        for po in pipe_res.all_poles:
            # Map to Hardware Mask with 5px buffer
            x1, y1, x2, y2 = [int(v) for v in po.box]
            cv2.rectangle(hardware_mask, (max(0, x1-5), max(0, y1-5)), (min(w, x2+5), min(h, y2+5)), 255, -1)
            
            final_detections.append({
                "label": "pole",
                "confidence": float(po.detection_conf),
                "bbox": [int(x) for x in po.box],
                "polygon": po.obb_polygon if hasattr(po, 'obb_polygon') else None,
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
                "confidence": float(conf),
                "bbox": [int(x) for x in box],
                "polygon": poly,
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

        # --- Wire Discovery Phase 2: Exclude static hardware ---
        # Any wire detected INSIDE a hardware box is disqualified to reduce noise
        mask_final = cv2.bitwise_and(mask_closed, cv2.bitwise_not(hardware_mask))
        
        # B. Generate Conductor Polygons from clean mask
        contours, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            cx, cy, cw, ch = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            
            # --- 1. Basic Size Filter ---
            if cw + ch < 80 or area < 100: 
                continue
            
            # --- 2. Geometric "Clump" Filter ---
            # Real wires are elongated. Huge square clumps are usually noise or shadows.
            aspect_ratio = max(cw, ch) / max(1, min(cw, ch))
            solidity = area / (cw * ch)
            
            # If it's a large clumpy rectangle (high solidity, low elongation), it's likely noise
            if cw > 150 and ch > 150 and solidity > 0.5 and aspect_ratio < 1.8:
                continue
            
            # --- 3. Process Valid Wire ---
            epsilon = 0.01 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            polygon = [[int(pt[0][0]), int(pt[0][1])] for pt in approx]
            
            # Localized thickness
            c_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(c_mask, [c], -1, 255, -1)
            local_skel = skel & (c_mask > 0)
            local_thickness = dist[local_skel > 0] * 2
            avg_thick = float(np.mean(local_thickness)) if len(local_thickness) > 0 else 0.0
            
            # Reject if the detected "wire" is physically impossible (too thick)
            if avg_thick > 80:
                continue
            
            final_detections.append({
                "label": "conductor",
                "confidence": 0.90,
                "bbox": [cx, cy, cx+cw, cy+ch],
                "polygon": polygon, # High precision UNet polygon
                "thickness": round(avg_thick, 1)
            })

        # Prepare Master Data (Asset Identity)
        master_data = {
            "final_class": pipe_res.final_class,
            "voltage": pipe_res.voltage,
            "reason": pipe_res.reason,
            "confidence": pipe_res.confidence,
            "pole_lean_angle": pipe_res.pole_orientation.lean_angle_deg if pipe_res.pole_orientation else 0.0,
            "pole_type": pipe_res.pole_orientation.pole_type if pipe_res.pole_orientation else "none",
            "pole_status": pipe_res.pole_orientation.fault_severity if pipe_res.pole_orientation else "none"
        }

        # Encode for response
        _, buffer = cv2.imencode('.jpg', img)
        img_b64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "detections": final_detections,
            "master": master_data,
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
    
    img_b64 = data['image'].split(',')[1] if ',' in data['image'] else data['image']
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
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files['image']
    try:
        result = process_image_file(file)
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Inference Error: {str(e)}"}), 500

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
        
        if isinstance(a_class, (dict, list)): a_class = json.dumps(a_class)
        if isinstance(a_volt, (dict, list)):  a_volt = json.dumps(a_volt)
        if isinstance(a_reason, (dict, list)): a_reason = json.dumps(a_reason)

        conn.execute('''
            INSERT INTO assets (id, worker_name, status, timestamp, asset_class, voltage, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (str(asset_id), str(worker_name), 'pending', str(timestamp), 
              str(a_class) if a_class is not None else None,
              str(a_volt) if a_volt is not None else None,
              str(a_reason) if a_reason is not None else None))
        
        # 2. Save Images
        for idx, img_data in enumerate(data['images']):
            # Ensure detections is properly serialized
            dets = img_data['detections']
            det_str = json.dumps(dets) if not isinstance(dets, str) else dets
            
            # DEEP LOGGING: Image
            print(f"[DB_LOG] save_asset Image[{idx}]: b64_len={len(img_data['image_b64']) if img_data.get('image_b64') else 'NONE'}, dets_len={len(det_str)}")

            # Clean B64 data (Strip prefix if exists)
            raw_b64 = str(img_data.get('image_b64', ''))
            if ',' in raw_b64: raw_b64 = raw_b64.split(',')[1]

            conn.execute('''
                INSERT INTO asset_images (asset_id, image_b64, detections, pole_angle)
                VALUES (?, ?, ?, ?)
            ''', (str(asset_id), raw_b64, det_str, float(img_data.get('pole_angle', 0.0))))
            
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
    # Get the first image of the asset
    row = conn.execute('SELECT image_b64, detections FROM asset_images WHERE asset_id = ? LIMIT 1', (asset_id,)).fetchone()
    conn.close()
    
    if not row:
        return "Asset image not found", 404
        
    from report_generator import annotate_image
    annotated_b64 = annotate_image(row['image_b64'], json.loads(row['detections'] or '[]'))
    
    # Ensure any prefix is stripped before final decode
    img_data = base64.b64decode(clean_b64(annotated_b64))
    buffer = io.BytesIO(img_data)
    
    return send_file(
        buffer,
        mimetype='image/jpeg',
        as_attachment=True,
        download_name=f"Annotated_Asset_{asset_id[:8]}.jpg"
    )
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

    # Get all activity logs for this asset
    logs = conn.execute('''
        SELECT user_name, action, details, timestamp
        FROM activity_logs 
        WHERE details LIKE ? 
        ORDER BY timestamp ASC
    ''', (f'%{asset_id}%',)).fetchall()

    # Get asset meta
    asset_row = conn.execute('SELECT * FROM assets WHERE id = ?', (asset_id,)).fetchone()

    # Get all images with current detections for the visual timeline
    image_rows = conn.execute(
        'SELECT image_b64, detections FROM asset_images WHERE asset_id = ? ORDER BY id ASC',
        (asset_id,)
    ).fetchall()

    conn.close()

    images_data = []
    for row in image_rows:
        try:
            dets = json.loads(row['detections']) if row['detections'] else []
        except:
            dets = []
        images_data.append({
            'image_b64': row['image_b64'],
            'detections': dets
        })

    return jsonify({
        'logs': [dict(l) for l in logs],
        'images': images_data,
        'asset_class': asset_row['asset_class'] if asset_row else '',
        'worker_name': asset_row['worker_name'] if asset_row else '',
        'status': asset_row['status'] if asset_row else '',
        'submitted_at': asset_row['timestamp'] if asset_row else ''
    })


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
    app.run(host='0.0.0.0', port=5000, debug=True)