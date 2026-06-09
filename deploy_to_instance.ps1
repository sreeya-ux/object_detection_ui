# Deployment script for object_detection_ui-main
# ===============================================

$SERVER_USER = "ubuntu"
$SERVER_HOST = "192.168.20.212"
$SERVER_PATH = "/home/ubuntu/object_detection_ui" # Adjust if path is different on server

Write-Host "--- Syncing Optimized Code to Instance ($SERVER_HOST) ---" -ForegroundColor Cyan

# List of all essential files for the new pipeline
$Files = @(
    "app.py",
    "worker.py",
    "pipeline.py",
    "config.py",
    "training_pipeline.py",
    "ocr_utils.py",
    "init_pg_db.py",
    "insulator_classifier.py",
    "rule_engine.py",
    "report_generator.py",
    "requirements.txt",
    "channels.yaml",
    "static/script.js",
    "templates/index.html",
    "templates/admin.html",
    "templates/login.html",
    "templates/asset_detail.html"
)

foreach ($file in $Files) {
    Write-Host "Uploading $file..." -ForegroundColor Yellow
    scp -i C:\Users\ASK037-PC\.ssh\id_ed25519_temp $file ${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}/$file
}

Write-Host "Uploading Models (this may take a moment)..." -ForegroundColor Yellow
scp -i C:\Users\ASK037-PC\.ssh\id_ed25519_temp -r models ${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}/
scp -i C:\Users\ASK037-PC\.ssh\id_ed25519_temp -r backup_models ${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}/

Write-Host "--- Updating Remote Environment ---" -ForegroundColor Cyan
# This will install requirements, sync DB, and restart the server manually (since systemd is missing)
ssh -i C:\Users\ASK037-PC\.ssh\id_ed25519_temp ${SERVER_USER}@${SERVER_HOST} "cd ${SERVER_PATH} && ./venv/bin/pip install -r requirements.txt && ./venv/bin/python init_pg_db.py && sudo systemctl restart infrastructure_ui"

Write-Host "--- Deployment & Database Sync Complete! Check your browser. ---" -ForegroundColor Green
