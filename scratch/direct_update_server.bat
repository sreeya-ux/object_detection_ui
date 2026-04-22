@echo off
setlocal

:: Configuration
set SERVER_IP=192.168.20.212
set USER=ubuntu
set REMOTE_DIR=/home/ubuntu/object_detection_ui
set SSH_KEY=C:\Users\ASK037-PC\.ssh\id_ed25519
set ASKPASS_PATH=c:\Users\ASK037-PC\Documents\object_detection_ui\scratch\askpass.bat

:: Set up AskPass for automated passphrase entry
set SSH_ASKPASS=%ASKPASS_PATH%
set GIT_ASKPASS=%ASKPASS_PATH%
set DISPLAY=:0

echo --- [1/3] TRANSFERRING FILES TO SERVER (%SERVER_IP%) ---
:: We only transfer the core files that changed
scp -o StrictHostKeyChecking=no app.py config.py crossarm_classifier.py pipeline.py report_generator.py init_db.py %USER%@%SERVER_IP%:%REMOTE_DIR%/
scp -o StrictHostKeyChecking=no scratch/patch_users.py %USER%@%SERVER_IP%:%REMOTE_DIR%/scratch/
scp -o StrictHostKeyChecking=no -r static/script.js %USER%@%SERVER_IP%:%REMOTE_DIR%/static/
scp -o StrictHostKeyChecking=no -r templates/asset_detail.html templates/index.html templates/admin.html %USER%@%SERVER_IP%:%REMOTE_DIR%/templates/

echo --- [2/3] MIGRATING DATABASE ON SERVER ---
ssh -o StrictHostKeyChecking=no %USER%@%SERVER_IP% "cd %REMOTE_DIR% && venv/bin/python3 scratch/patch_users.py"

echo --- [3/3] RESTARTING SERVICE ---
ssh -o StrictHostKeyChecking=no %USER%@%SERVER_IP% "sudo systemctl restart infrastructure_ui"

echo --- DEPLOYMENT COMPLETE ---
pause
