# 🌐 Asakta Vision AI Custom Domain & Hosting Guide

This guide provides step-by-step instructions to link your purchased domain **`surveyai.live`** (from Namecheap) to this Flask-based machine learning application.

Because this application uses heavy PyTorch models (`FastSAM`, YOLO segment, UNet), hosting it on a standard low-cost cloud VM can run out of memory or execute very slowly without a GPU. Therefore, we provide **three highly optimized ways** to host your app, ranging from a completely free, local GPU-accelerated tunnel to a traditional Cloud VPS server.

---

## 🛠️ Step 1: Configure Your Namecheap DNS Settings

No matter which hosting option you select below, you will need to configure DNS records in your Namecheap control panel.

1. **Log in** to your Namecheap account at [Namecheap.com](https://www.namecheap.com/).
2. Navigate to your **Domain List** in the left sidebar and click **Manage** next to `surveyai.live`.
3. Select the **Advanced DNS** tab at the top.
4. Locate the **Host Records** section. Here you will add the DNS records depending on your chosen hosting method below.

---

## ⚡ Option 1: Cloudflare Tunnel (Highly Recommended & 100% Free)

> [!TIP]
> **Why Cloudflare Tunnels?**
> A Cloudflare Tunnel (`cloudflared`) connects your local development machine (which has your GPU, model weights, and setup) directly to the Cloudflare edge network. This allows you to host the app **locally on your high-performance hardware** while serving it safely over the internet at `https://surveyai.live` with a free SSL certificate! It is completely free, secure, and doesn't require port forwarding.

### Step 1.1: Point Namecheap to Cloudflare Nameservers
1. Sign up for a free account at [Cloudflare](https://dash.cloudflare.com/).
2. Click **Add a Site** and enter `surveyai.live`.
3. Choose the **Free Plan**.
4. Cloudflare will scan your records and provide two custom nameservers (e.g., `aria.ns.cloudflare.com` and `will.ns.cloudflare.com`).
5. Go back to your **Namecheap Domain Details** page under the **Domain** tab.
6. Scroll to **Nameservers**, change the dropdown to **Custom DNS**, enter the two Cloudflare Nameservers, and click the green checkmark to save.
   *(Note: DNS propagation can take from 10 minutes to a few hours).*

### Step 1.2: Set Up the Cloudflare Tunnel on Your Machine
1. Install Cloudflare's daemon (`cloudflared`) on your hosting machine:
   * **Windows (PowerShell as Admin):**
     ```powershell
     winget install --id Cloudflare.cloudflared
     ```
   * **Mac (Homebrew):**
     ```bash
     brew install cloudflared
     ```
2. Log in and authorize your domain:
   ```bash
   cloudflared tunnel login
   ```
   *(A browser window will open. Select `surveyai.live` and click **Authorize**).*

3. Create your secure tunnel (name it `asakta-vision`):
   ```bash
   cloudflared tunnel create asakta-vision
   ```
   *Take note of the Tunnel ID outputted in your console (looks like `a1b2c3d4-e5f6-...`).*

4. Route the tunnel to your domain:
   ```bash
   cloudflared tunnel route dns asakta-vision surveyai.live
   ```

5. Configure the local routing rules. Create a file named `config.yml` in your local Cloudflare directory (usually `C:\Users\<username>\.cloudflared\config.yml` on Windows):
   ```yaml
   tunnel: <YOUR_TUNNEL_ID>
   credentials-file: C:\Users\<username>\.cloudflared\<YOUR_TUNNEL_ID>.json

   ingress:
     - hostname: surveyai.live
       service: http://localhost:5002
     - hostname: www.surveyai.live
       service: http://localhost:5002
     - service: http_status:404
   ```

6. **Start the tunnel:**
   ```bash
   cloudflared tunnel run asakta-vision
   ```
   Your app is now securely hosted and live at **`https://surveyai.live`**!

---

## 🚀 Option 2: Ngrok Custom Domain Tunneling

If you already have a paid Ngrok subscription, you can run Ngrok directly with your custom domain.

### Step 2.1: Add a CNAME Record in Namecheap
1. Go to your **Namecheap Advanced DNS** tab.
2. Click **Add New Record**:
   * **Type:** `CNAME Record`
   * **Host:** `@` (or a subdomain like `app`)
   * **Value:** `<your-custom-cname-target>.ngrok-free.app` *(Ngrok will provide this when you set up a domain in your dashboard).*
   * **TTL:** `Automatic`

### Step 2.2: Run Ngrok Locally
Launch the tunnel using your custom domain pointing to your Flask local port:
```bash
ngrok http 5002 --domain=surveyai.live
```

---

## ☁️ Option 3: Dedicated Cloud Server / VPS (AWS, DigitalOcean, GCP)

If you wish to deploy the application on a permanently running Linux VPS, follow these production setup instructions.

### Step 3.1: Configure Namecheap DNS
Add an A Record pointing directly to your VPS public IP:
1. Go to the **Namecheap Advanced DNS** tab.
2. Under **Host Records**, click **Add New Record**:
   * **Type:** `A Record`
   * **Host:** `@`
   * **Value:** `<YOUR_VPS_PUBLIC_IP>` (e.g., `192.0.2.1`)
   * **TTL:** `Automatic`
3. Add a CNAME Record for the `www` subdomain:
   * **Type:** `CNAME Record`
   * **Host:** `www`
   * **Value:** `surveyai.live`
   * **TTL:** `Automatic`

### Step 3.2: Set Up Server Dependencies
Connect to your Ubuntu VPS via SSH and install the system prerequisites:
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv git nginx libgl1-mesa-glx libglib2.0-0 -y
```

### Step 3.3: Clone and Setup Virtual Environment
```bash
git clone <YOUR_GIT_REPO_URL> /var/www/object_detection_ui
cd /var/www/object_detection_ui
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python init_db.py
```

### Step 3.4: Setup Systemd Service
Create a systemd service file to keep Gunicorn running in the background:
```bash
sudo nano /etc/systemd/system/asakta.service
```

Add the following configuration:
```ini
[Unit]
Description=Asakta Vision AI Flask Application
After=network.target

[Service]
User=root
WorkingDirectory=/var/www/object_detection_ui
Environment="PATH=/var/www/object_detection_ui/venv/bin"
ExecStart=/var/www/object_detection_ui/venv/bin/gunicorn --workers 3 --bind 127.0.0.1:5002 --timeout 120 app:app

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl enable asakta
sudo systemctl start asakta
```

### Step 3.5: Configure Nginx as a Reverse Proxy
Create an Nginx server block to handle traffic and route it to Gunicorn:
```bash
sudo nano /etc/nginx/sites-available/asakta
```

Add the configuration below:
```nginx
server {
    listen 80;
    server_name surveyai.live www.surveyai.live;

    location / {
        proxy_pass http://127.0.0.1:5002;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        client_max_body_size 50M; # Accommodates large image uploads
    }

    # Serve static assets directly for maximum performance
    location /static/ {
        alias /var/www/object_detection_ui/static/;
    }
}
```

Enable the Nginx block and restart:
```bash
sudo ln -s /etc/nginx/sites-available/asakta /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### Step 3.6: Install SSL (HTTPS) with Let's Encrypt
```bash
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d surveyai.live -d www.surveyai.live
```
Follow the interactive prompts to automatically obtain and configure the SSL certificates. Certbot will set up automatic 90-day renewals!

---

## 🔒 Post-Deployment Security Best Practices

1. **Secret Key**: In production, change `app.secret_key` inside `app.py` to a highly secure random string:
   ```python
   app.secret_key = os.environ.get("FLASK_SECRET_KEY", "your-long-secure-random-string-here")
   ```
2. **Database Backup**: Regularly back up `database.db` (if using SQLite) or migrate to a managed PostgreSQL cluster in production (pre-configured in `config.py`!).
