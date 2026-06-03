import os
import json
import requests
import cv2
import numpy as np
from tqdm import tqdm

# =========================
# CONFIG
# =========================
NDJSON_PATH = r"C:\Users\ASK037-PC\Downloads\Export  project - channels,insutaors,conductors2 - 5_12_2026 (2).ndjson"
OUTPUT_DIR = "dataset_channels"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbXAyNnY2ZTcwNXpyMDd5ZTJueWM3anJjIiwib3JnYW5pemF0aW9uSWQiOiJjbXAyNnY2ZHgwNXpxMDd5ZWdjMGhieHNsIiwiYXBpS2V5SWQiOiJjbXBiMDY5dWEwc2RrMDcyeTFqdnhlOWdyIiwic2VjcmV0IjoiODQ5NzRhM2RmZTc5MDU4M2ExZWYwYzljYzEzYWUyNjciLCJpYXQiOjE3NzkwOTY1ODksImV4cCI6MTc4NzU2Mzc4OX0.oUPYg3yf1GYnDst19A_8cU08GQcEVXXDFttksFr2zYo"

HEADERS = {"Authorization": f"Bearer {API_KEY}"}

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

# =========================
# PROCESS
# =========================
with open(NDJSON_PATH, "r") as f:
    for idx, line in enumerate(tqdm(f)):
        if not line.strip(): continue
        try:
            item = json.loads(line)
            image_url = item["data_row"]["row_data"]
            img_name = f"img_{idx}.jpg"
            img_path = os.path.join(OUTPUT_DIR, "images", "train", img_name)
            
            # Step 1: Get Image (Download or Read)
            image = None
            if os.path.exists(img_path):
                image = cv2.imread(img_path)
            
            if image is None:
                # Use HEADERS for image download
                img_resp = requests.get(image_url, headers=HEADERS)
                if img_resp.status_code == 200:
                    nparr = np.frombuffer(img_resp.content, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if image is not None:
                        cv2.imwrite(img_path, image)
            
            if image is None: 
                print(f"   [SKIP] Could not download image {idx}")
                continue
                
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
                                    # Convert to 4-pt polygon for segmentation consistency
                                    pts = [cx-bw/2, cy-bh/2, cx+bw/2, cy-bh/2, cx+bw/2, cy+bh/2, cx-bw/2, cy+bh/2]
                                    label_lines.append(f"{class_id} " + " ".join(map(str, pts)))

            if label_lines:
                label_path = os.path.join(OUTPUT_DIR, "labels", "train", img_name.replace(".jpg", ".txt"))
                with open(label_path, "w") as f_lbl:
                    f_lbl.write("\n".join(label_lines))

        except Exception as e:
            print(f"Error {idx}: {e}")

print("Dataset ready for training.")