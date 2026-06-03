import os
import json
import shutil
import requests
import cv2
from tqdm import tqdm
from labelbox import Client

# ==========================================
# CONFIGURATION
# ==========================================
OUTPUT_DIR = "dataset_combined"

# Set 1 paths (603 images)
SET1_IMG_DIR = r"dataset_channels\images\train"
SET1_LBL_DIR = r"dataset_channels\labels\train"

# Set 2 paths (700 images)
SET2_NDJSON = r"C:\Users\ASK037-PC\Downloads\Export  project - channels,insutaors,conductors2 - 5_12_2026 (2).ndjson"
SET2_IMG_SRC = r"C:\Users\ASK037-PC\Downloads\rdss_gis_images\rdss_gis_images"

# Set 3 (959 images live)
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbXAyNnY2ZTcwNXpyMDd5ZTJueWM3anJjIiwib3JnYW5pemF0aW9uSWQiOiJjbXAyNnY2ZHgwNXpxMDd5ZWdjMGhieHNsIiwiYXBpS2V5SWQiOiJjbXBiMDY5dWEwc2RrMDcyeTFqdnhlOWdyIiwic2VjcmV0IjoiODQ5NzRhM2RmZTc5MDU4M2ExZWYwYzljYzEzYWUyNjciLCJpYXQiOjE3NzkwOTY1ODksImV4cCI6MTc4NzU2Mzc4OX0.oUPYg3yf1GYnDst19A_8cU08GQcEVXXDFttksFr2zYo"

CLASS_MAP_SET2 = {
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

CLASS_MAP_SET3 = {
    "Insulators": 0,
    "V Cross Arm": 1,
    "Tapping": 2,
    "Top Cleat": 3,
    "Side Arm": 4,
    "T rising": 5,
    "Special Clamp": 6,
    "Street Light": 7,
    "Stay Set": 8,
    "Box Arm": 9,
    "AB Switch": 10,
    "DTR": 11
}

# Create target directories
os.makedirs(os.path.join(OUTPUT_DIR, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labels", "train"), exist_ok=True)


def parse_and_convert_labelbox_item(item, class_map, w, h):
    """Parses objects from a Labelbox export item and returns YOLO format lines."""
    label_lines = []
    for p_id in item.get("projects", {}):
        for label in item["projects"][p_id].get("labels", []):
            if "annotations" in label and "objects" in label["annotations"]:
                for obj in label["annotations"]["objects"]:
                    val = obj.get("value")
                    if val in class_map:
                        class_id = class_map[val]
                        if "polygon" in obj and obj["polygon"]:
                            coords = [f"{pt['x']/w} {pt['y']/h}" for pt in obj["polygon"]]
                            label_lines.append(f"{class_id} " + " ".join(coords))
                        elif "bounding_box" in obj and obj["bounding_box"]:
                            box = obj["bounding_box"]
                            cx, cy = (box["left"] + box["width"]/2)/w, (box["top"] + box["height"]/2)/h
                            bw, bh = box["width"]/w, box["height"]/h
                            pts = [cx-bw/2, cy-bh/2, cx+bw/2, cy-bh/2, cx+bw/2, cy+bh/2, cx-bw/2, cy+bh/2]
                            label_lines.append(f"{class_id} " + " ".join(map(str, pts)))
    return label_lines


# ==========================================
# PROCESS SET 1 (603 IMAGES)
# ==========================================
print("Processing Set 1 (603 images from disk)...")
set1_count = 0
if os.path.exists(SET1_IMG_DIR):
    for filename in os.listdir(SET1_IMG_DIR):
        if filename.endswith(".jpg"):
            base = filename.replace(".jpg", "")
            img_src = os.path.join(SET1_IMG_DIR, filename)
            lbl_src = os.path.join(SET1_LBL_DIR, base + ".txt")
            
            img_dst = os.path.join(OUTPUT_DIR, "images", "train", f"set1_{filename}")
            lbl_dst = os.path.join(OUTPUT_DIR, "labels", "train", f"set1_{base}.txt")
            
            shutil.copy2(img_src, img_dst)
            if os.path.exists(lbl_src):
                shutil.copy2(lbl_src, lbl_dst)
            set1_count += 1
print(f"Copied {set1_count} images for Set 1.")

# ==========================================
# PROCESS SET 2 (700 IMAGES)
# ==========================================
print("\nProcessing Set 2 (700 images from local NDJSON & rdss)...")
set2_count = 0
set2_data = []
with open(SET2_NDJSON, "r") as f:
    for line in f:
        if line.strip():
            set2_data.append(json.loads(line))

for idx, item in enumerate(tqdm(set2_data)):
    try:
        ext_id = item["data_row"]["external_id"]
        img_src_path = os.path.join(SET2_IMG_SRC, ext_id)
        
        if not os.path.exists(img_src_path):
            continue
            
        img_dst_path = os.path.join(OUTPUT_DIR, "images", "train", f"set2_{ext_id}")
        shutil.copy2(img_src_path, img_dst_path)
        
        # Load image to get width and height for coordinate normalization
        image = cv2.imread(img_dst_path)
        if image is None:
            continue
        h, w = image.shape[:2]
        
        label_lines = parse_and_convert_labelbox_item(item, CLASS_MAP_SET2, w, h)
        if label_lines:
            lbl_dst_path = os.path.join(OUTPUT_DIR, "labels", "train", f"set2_{ext_id.replace('.jpg', '.txt')}")
            with open(lbl_dst_path, "w") as f_lbl:
                f_lbl.write("\n".join(label_lines))
            set2_count += 1
            
    except Exception as e:
        print(f"Error Set 2 idx {idx}: {e}")
print(f"Processed {set2_count} images for Set 2.")

# ==========================================
# PROCESS SET 3 (959 IMAGES LIVE)
# ==========================================
print("\nConnecting to Labelbox for Set 3 (Live)...")
client = Client(api_key=API_KEY)
project = [p for p in client.get_projects() if p.name == 'components3'][0]
print(f"Exporting live project '{project.name}'...")

# Optimize params to only fetch required details, avoiding timeouts/disconnects
export_params = {
    "attachments": False,
    "embeddings": False,
    "metadata_fields": False,
    "project_details": False,
    "reviews": False,
    "collaborators": False,
    "label_details": True,
    "data_row_details": True,
    "performance_details": False
}

import time
max_retries = 5
for attempt in range(max_retries):
    try:
        export_task = project.export(params=export_params)
        export_task.wait_till_done()
        stream = export_task.get_buffered_stream()
        set3_items = list(stream)
        break
    except Exception as e:
        print(f"Attempt {attempt + 1} failed: {e}")
        if attempt == max_retries - 1:
            raise e
        print("Waiting 10 seconds before retrying...")
        time.sleep(10)

print(f"Found {len(set3_items)} live records. Syncing...")


from concurrent.futures import ThreadPoolExecutor, as_completed

def process_set3_item(item_data):
    try:
        item = item_data.json
        data_row_id = item["data_row"]["id"]
        img_name = f"set3_{data_row_id}.jpg"
        img_path = os.path.join(OUTPUT_DIR, "images", "train", img_name)
        
        # Get/download image
        image = None
        if os.path.exists(img_path) and os.path.getsize(img_path) > 5000:
            image = cv2.imread(img_path)
            
        if image is None:
            image_url = item["data_row"]["row_data"]
            img_data = requests.get(image_url).content
            with open(img_path, "wb") as f_img:
                f_img.write(img_data)
            image = cv2.imread(img_path)
            
        if image is None:
            return False
            
        h, w = image.shape[:2]
        
        label_lines = parse_and_convert_labelbox_item(item, CLASS_MAP_SET3, w, h)
        if label_lines:
            lbl_path = os.path.join(OUTPUT_DIR, "labels", "train", img_name.replace(".jpg", ".txt"))
            with open(lbl_path, "w") as f_lbl:
                f_lbl.write("\n".join(label_lines))
            return True
    except Exception as e:
        print(f"Error processing item: {e}")
    return False

print("\nStarting parallel download and processing of Set 3...")
set3_count = 0
with ThreadPoolExecutor(max_workers=20) as executor:
    futures = {executor.submit(process_set3_item, item): item for item in set3_items}
    for future in tqdm(as_completed(futures), total=len(futures)):
        if future.result():
            set3_count += 1

print(f"Processed {set3_count} images for Set 3.")

print("\n===========================================")
print("COMBINATION COMPLETE!")
print(f"Total set1 images: {set1_count}")
print(f"Total set2 images: {set2_count}")
print(f"Total set3 images: {set3_count}")
print(f"Combined dataset output folder: {OUTPUT_DIR}")
print("===========================================")
