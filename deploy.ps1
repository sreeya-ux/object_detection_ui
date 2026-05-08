# Deployment Script - Run this after editing variables
# ================================================

# EDIT THESE VARIABLES:
$SERVER_USER = "ubuntu"
$SERVER_HOST = "192.168.20.212"
$SERVER_PATH = "/home/ubuntu/object_detection_ui"
$SERVICE_NAME = "infrastructure_ui.service"

Write-Host "=== Deploying object_detection_ui ===" -ForegroundColor Cyan

# Step 1: Git operations
Write-Host "`n[1/4] Pushing to GitHub..." -ForegroundColor Yellow
git add -A
git commit -m "Update: Enhanced pole detection and silhouette filtering" 2>$null
$pushResult = git push origin main 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Push failed. Check your git remote and credentials." -ForegroundColor Red
    Write-Host $pushResult
    exit 1
}
Write-Host "Push successful!" -ForegroundColor Green

# Step 2: SSH to server and pull
Write-Host "`n[2/4] Connecting to server and pulling changes..." -ForegroundColor Yellow
$sshCmd = "cd $SERVER_PATH && git pull origin main"
ssh ${SERVER_USER}@${SERVER_HOST} $sshCmd
if ($LASTEXITCODE -ne 0) {
    Write-Host "Server pull failed. Check SSH access and server path." -ForegroundColor Red
    exit 1
}

# Step 3: Restart service
Write-Host "`n[3/4] Restarting service..." -ForegroundColor Yellow
$restartCmd = "sudo systemctl restart $SERVICE_NAME"
ssh ${SERVER_USER}@${SERVER_HOST} $restartCmd
if ($LASTEXITCODE -ne 0) {
    Write-Host "Service restart failed. Check service name." -ForegroundColor Red
    exit 1
}

# Step 4: Verify
Write-Host "`n[4/4] Verifying service status..." -ForegroundColor Yellow
$statusCmd = "sudo systemctl status $SERVICE_NAME --no-pager"
ssh ${SERVER_USER}@${SERVER_HOST} $statusCmd

Write-Host "`n=== Deployment Complete ===" -ForegroundColor Green
