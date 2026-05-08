import os
import json
import requests
import cv2
import numpy as np
from tqdm import tqdm

# =========================
# CONFIG
# =========================
NDJSON_PATH = r"C:\Users\ASK037-PC\Downloads\Export  project - only_pole_2 - 5_4_2026.ndjson"
OUTPUT_DIR = "dataset"

API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbW9sM3doa3QwNmxoMDd6aGd5aDVjenl5Iiwib3JnYW5pemF0aW9uSWQiOiJjbW9sM3doa2owNmxnMDd6aGFxY2Vkc3RjIiwiYXBpS2V5SWQiOiJjbW9xcWw4ZHMwMXZtMDd6ZzJjcDY4amthIiwic2VjcmV0IjoiNmVjNWIwMTY4ZDk5YWY2ZGFmMTJhYzY1NWE0YzJlMjYiLCJpYXQiOjE3Nzc4NzExNjcsImV4cCI6MTc4MDI5MDM2N30.pXNmsAwOEnKeshJPHT0qJ3DKh3AcA_FskN7iv8SpXDI"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

CLASS_MAP = {
    "main_pole": 0,
    "strut_pole": 1
}

# =========================
# CREATE FOLDERS
# =========================
os.makedirs(f"{OUTPUT_DIR}/images/train", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/labels/train", exist_ok=True)

# =========================
# READ NDJSON
# =========================
data = []
with open(NDJSON_PATH, "r") as f:
    for line in f:
        data.append(json.loads(line))

PROJECT_ID = list(data[0]["projects"].keys())[0]

# =========================
# PROCESS
# =========================
for idx, item in enumerate(tqdm(data)):
    try:
        image_url = item["data_row"]["row_data"]

        labels_data = item["projects"][PROJECT_ID]["labels"]
        if not labels_data:
            continue

        objects = labels_data[0]["annotations"]["objects"]

        # DOWNLOAD IMAGE
        img_data = requests.get(image_url).content
        img_name = f"img_{idx}.jpg"
        img_path = f"{OUTPUT_DIR}/images/train/{img_name}"

        with open(img_path, "wb") as f:
            f.write(img_data)

        image = cv2.imread(img_path)
        h, w = image.shape[:2]

        label_lines = []

        for obj in objects:
            class_name = obj["value"]

            if class_name not in CLASS_MAP:
                continue

            class_id = CLASS_MAP[class_name]

            mask_url = obj["mask"]["url"]

            # 🔥 FIX: use headers here
            mask_data = requests.get(mask_url, headers=HEADERS).content

            mask = np.frombuffer(mask_data, np.uint8)
            mask = cv2.imdecode(mask, cv2.IMREAD_GRAYSCALE)

            if mask is None:
                continue

            _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                if len(cnt) < 5:
                    continue

                cnt = cnt.reshape(-1, 2)

                yolo_coords = []
                for x, y in cnt:
                    yolo_coords.append(x / w)
                    yolo_coords.append(y / h)

                line = str(class_id) + " " + " ".join(map(str, yolo_coords))
                label_lines.append(line)

        if label_lines:
            label_path = f"{OUTPUT_DIR}/labels/train/{img_name.replace('.jpg','.txt')}"
            with open(label_path, "w") as f:
                f.write("\n".join(label_lines))

    except Exception as e:
        print(f"[ERROR] {idx}: {e}")

print("✅ Done!")