from flask import Flask, render_template, request, jsonify, session, redirect, url_for, make_response
from werkzeug.security import check_password_hash
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
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def get_ngrok_url():
    try:
        response = requests.get('http://127.0.0.1:4040/api/tunnels', timeout=0.1)
        if response.status_code == 200:
            tunnels = response.json().get('tunnels', [])
            for tunnel in tunnels:
                if tunnel.get('proto') == 'https':
                    return tunnel.get('public_url')
    except:
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
                if user['role'] == 'admin':
                    return redirect(url_for('admin_dashboard'))
                return redirect(url_for('home'))
        
        return render_template('login.html', error="Invalid credentials", ngrok_url=get_ngrok_url())
    
    return render_template('login.html', ngrok_url=get_ngrok_url())

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# =========================
# IMAGE PROCESSING
# =========================
def process_image_file(file_stream):
    """
    Main diagnostic entry point.
    Combines Rule Engine (InfrastructurePipeline) with UNet Conductor Segmentation.
    """
    # Create a temporary file to run the pipeline.predict (which expects a path)
    temp_filename = f"temp_{uuid.uuid4()}.jpg"
    with open(temp_filename, "wb") as f:
        f.write(file_stream.read())
    
    try:
        # 1. Run the Rule Engine Pipeline
        # This identifies structural components, classifies crossarms, and runs the master logic.
        pipe_res = pipeline_engine.predict(temp_filename, visualize=False)
        
        # Reload image for UNet processing and base64 response
        img = cv2.imread(temp_filename)
        h, w = img.shape[:2]
        
        # 2. Process Conductors with UNet Segmentation Model (ResNet34)
        input_img = cv2.resize(img, (512, 512)).transpose(2, 0, 1) / 255.0
        tensor = torch.tensor(input_img[None, ...], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            out = unet_model(tensor)
            mask = torch.sigmoid(out).squeeze().cpu().numpy()
            
        mask_binary = (mask > 0.25).astype(np.uint8) * 255
        mask_resized = cv2.resize(mask_binary, (w, h), interpolation=cv2.INTER_NEAREST)

        # Thickness Measurement via Distance Transform & Skeletonize
        dist = cv2.distanceTransform(mask_resized, cv2.DIST_L2, 5)
        from skimage.morphology import skeletonize
        skel = (skeletonize(mask_resized / 255.0) > 0).astype(np.uint8)

        # Bridge gaps for continuous polygons (Smaller kernel to preserve detail)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        mask_closed = cv2.morphologyEx(mask_resized, cv2.MORPH_CLOSE, kernel)

        final_detections = []

        # A. Map Rule Engine Components to UI format (with OBB Polygons)
        # Each component in pipe_res is now (box, conf, angle, polygon)
        for ins in pipe_res.insulators:
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
            final_detections.append({
                "label": "crossarm",
                "confidence": float(ca.detection_conf),
                "bbox": [int(x) for x in ca.box],
                "polygon": ca.obb_polygon if hasattr(ca, 'obb_polygon') else None,
                "details": {
                    "shape": ca.shape
                }
            })
        
        if pipe_res.pole_orientation:
            po = pipe_res.pole_orientation
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

        # B. Generate Conductor Polygons from UNet (The high-precision part)
        contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            cx, cy, cw, ch = cv2.boundingRect(c)
            if cw + ch < 80: continue
            
            epsilon = 0.01 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            polygon = [[int(pt[0][0]), int(pt[0][1])] for pt in approx]
            
            # Localized thickness
            c_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(c_mask, [c], -1, 255, -1)
            local_skel = skel & (c_mask > 0)
            local_thickness = dist[local_skel > 0] * 2
            avg_thick = float(np.mean(local_thickness)) if len(local_thickness) > 0 else 0.0
                
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
            "confidence": pipe_res.confidence
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
        # Cleanup temporary image file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

# =========================
# FLASK ROUTES
# =========================
@app.route('/')
@login_required
def home():
    ngrok_url = get_ngrok_url()
    return render_template('index.html', ngrok_url=ngrok_url)

@app.route('/admin')
@admin_required
def admin_dashboard():
    return render_template('admin.html')

@app.route('/admin/export')
@admin_required
def export_tasks():
    conn = get_db_connection()
    tasks = conn.execute('SELECT * FROM tasks ORDER BY timestamp DESC').fetchall()
    conn.close()

    # Create CSV in memory
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(['TASK_ID', 'TIMESTAMP', 'WORKER', 'STATUS', 'LABEL', 'CONFIDENCE', 'X1', 'Y1', 'X2', 'Y2', 'MANUAL_ENTRY'])
    
    for task in tasks:
        try:
            detections = json.loads(task['detections'])
            for d in detections:
                bbox = d.get('bbox', [0,0,0,0])
                cw.writerow([
                    task['id'],
                    task['timestamp'],
                    task['worker_name'],
                    task['status'],
                    d.get('label', 'UNKNOWN'),
                    f"{float(d.get('confidence', 0)):.2f}",
                    bbox[0], bbox[1], bbox[2], bbox[3],
                    d.get('manual', False)
                ])
        except Exception as e:
            print(f"Error exporting task {task['id']}: {e}")
            continue
            
    output = make_response(si.getvalue())
    filename = f"asakta_report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    output.headers["Content-Disposition"] = f"attachment; filename={filename}"
    output.headers["Content-type"] = "text/csv"
    return output

@app.route('/admin/task/<task_id>')
@admin_required
def view_task(task_id):
    conn = get_db_connection()
    task_row = conn.execute('SELECT * FROM tasks WHERE id = ?', (task_id,)).fetchone()
    conn.close()
    
    if not task_row:
        return "Task not found", 404
        
    task = dict(task_row)
    detections = json.loads(task['detections'])
    
    grouped_detections = {}
    for det in detections:
        label = det['label']
        if label not in grouped_detections:
            grouped_detections[label] = []
    grouped_detections[label].append(det)
    
    task['grouped_detections'] = grouped_detections
    task['total_count'] = len(detections)
    return render_template('task_detail.html', task=task)

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
        return jsonify({"error": str(e)}), 500

# =========================
# API ENDPOINTS
# =========================
@app.route('/api/save_task', methods=['POST'])
@login_required
def save_task():
    data = request.json
    task_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    worker_name = session['user']
    
    conn = get_db_connection()
    conn.execute('''
        INSERT INTO tasks (id, worker_name, image_b64, detections, status, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (task_id, worker_name, data['image_b64'], json.dumps(data['detections']), 'pending', timestamp))
    conn.commit()
    conn.close()
    
    return jsonify({"status": "success", "task_id": task_id})

@app.route('/api/get_tasks')
@login_required
def get_tasks():
    status = request.args.get('status')
    
    conn = get_db_connection()
    query = 'SELECT * FROM tasks WHERE 1=1'
    params = []
    
    if status:
        query += ' AND status = ?'
        params.append(status)
    
    if session['role'] != 'admin':
        query += ' AND worker_name = ?'
        params.append(session['user'])
    
    tasks_rows = conn.execute(query, params).fetchall()
    conn.close()
    
    tasks = []
    for row in tasks_rows:
        task = dict(row)
        task['detections'] = json.loads(task['detections'])
        tasks.append(task)
        
    return jsonify(tasks)

@app.route('/api/update_task', methods=['POST'])
@login_required
def update_task():
    data = request.json
    task_id = data.get('task_id')
    status = data.get('status')
    
    conn = get_db_connection()
    conn.execute('UPDATE tasks SET status = ? WHERE id = ?', (status, task_id))
    conn.commit()
    conn.close()
    
    return jsonify({"status": "success"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)