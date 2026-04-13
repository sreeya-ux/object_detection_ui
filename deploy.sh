#!/bin/bash
# ---------------------------------------------------------
# Deploy Script for Infrastructure Detection UI (Ubuntu)
# ---------------------------------------------------------

echo "--- 1. Updating System Packages ---"
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv libgl1-mesa-glx libglib2.0-0

echo "--- 2. Setting Up Virtual Environment ---"
python3 -m venv venv
source venv/bin/bin/activate

echo "--- 3. Installing Dependencies (this may take a few minutes) ---"
pip install --upgrade pip
pip install -r requirements.txt

echo "--- 4. Initializing Database ---"
python3 init_db.py

echo "--- 5. Setting Up System Service ---"
sudo cp infrastructure_ui.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable infrastructure_ui
sudo systemctl restart infrastructure_ui

echo "--- Deployment Complete! ---"
echo "You can access the dashboard at: http://$(hostname -I | awk '{print $1}'):5000"
