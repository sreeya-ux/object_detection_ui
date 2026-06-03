import os
import json
import requests
import cv2
import numpy as np
from tqdm import tqdm
from labelbox import Client

# =========================
# CONFIG
# =========================
NDJSON_PATH = r"C:\Users\ASK037-PC\Downloads\Export  project - channels,insutaors,conductors2 - 5_12_2026 (2).ndjson"
OUTPUT_DIR = "dataset_channels"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbXAyNnY2ZTcwNXpyMDd5ZTJueWM3anJjIiwib3JnYW5pemF0aW9uSWQiOiJjbXAyNnY2ZHgwNXpxMDd5ZWdjMGhieHNsIiwiYXBpS2V5SWQiOiJjbXBiMDY5dWEwc2RrMDcyeTFqdnhlOWdyIiwic2VjcmV0IjoiODQ5NzRhM2RmZTc5MDU4M2ExZWYwYzljYzEzYWUyNjciLCJpYXQiOjE3NzkwOTY1ODksImV4cCI6MTc4NzU2Mzc4OX0.oUPYg3yf1GYnDst19A_8cU08GQcEVXXDFttksFr2zYo"

CLASS_MAP = {
    "insulators": 0,
    "v_cross_arm": 1,
    "tapping_arm": 2,
    "top_cleat": 3,
    "side_arm": 4,
    "t_rising": 5,
    "special_clamp": 6,
    "street_light": 7,
    "stay_set": 8,
    "box_arm": 9,
    "ab_switch": 10,
    "dtr": 11
}

os.makedirs(f"{OUTPUT_DIR}/images/train", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/labels/train", exist_ok=True)

# Initialize Labelbox Client
client = Client(api_key=API_KEY)

# =========================
# PROCESS
# =========================
with open(NDJSON_PATH, "r") as f:
    lines = [line for line in f if line.strip()]

print(f"Found {len(lines)} records. Refreshing URLs and downloading...")

for idx, line in enumerate(tqdm(lines)):
    try:
        item = json.loads(line)
        data_row_id = item["data_row"]["id"]
        img_name = f"{data_row_id}.jpg"
        img_path = os.path.join(OUTPUT_DIR, "images", "train", img_name)
        
        # Step 1: Get Fresh Image URL from SDK
        image = None
        # Always re-download if the existing file is small (forbidden error)
        if os.path.exists(img_path) and os.path.getsize(img_path) > 5000:
            image = cv2.imread(img_path)
        
        if image is None:
            # Fetch fresh URL using SDK
            data_row = client.get_data_row(data_row_id)
            fresh_url = data_row.row_data
            
            img_resp = requests.get(fresh_url)
            if img_resp.status_code == 200:
                nparr = np.frombuffer(img_resp.content, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if image is not None:
                    cv2.imwrite(img_path, image)
            else:
                # If SDK row_data fails, try using the direct image_url with HEADERS as fallback
                # but SDK is usually better.
                pass
        
        if image is None: continue
            
        h, w = image.shape[:2]

        # Step 2: Extract Labels
        label_lines = []
        for p_id in item.get("projects", {}):
            for label in item["projects"][p_id].get("labels", []):
                if "annotations" in label and "objects" in label["annotations"]:
                    for obj in label["annotations"]["objects"]:
                        val = obj.get("value")
                        if val in CLASS_MAP:
                            class_id = CLASS_MAP[val]
                            if "polygon" in obj:
                                coords = [f"{pt['x']/w} {pt['y']/h}" for pt in obj["polygon"]]
                                label_lines.append(f"{class_id} " + " ".join(coords))
                            elif "bounding_box" in obj:
                                box = obj["bounding_box"]
                                cx, cy = (box["left"] + box["width"]/2)/w, (box["top"] + box["height"]/2)/h
                                bw, bh = box["width"]/w, box["height"]/h
                                pts = [cx-bw/2, cy-bh/2, cx+bw/2, cy-bh/2, cx+bw/2, cy+bh/2, cx-bw/2, cy+bh/2]
                                label_lines.append(f"{class_id} " + " ".join(map(str, pts)))

        if label_lines:
            label_path = os.path.join(OUTPUT_DIR, "labels", "train", img_name.replace(".jpg", ".txt"))
            with open(label_path, "w") as f_lbl:
                f_lbl.write("\n".join(label_lines))

    except Exception as e:
        print(f"Error {idx}: {e}")

print("Dataset refreshed and ready for training!")
