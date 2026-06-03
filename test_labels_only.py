import os
import json
from tqdm import tqdm

# =========================
# CONFIG
# =========================
NDJSON_PATH = r"C:\Users\ASK037-PC\Downloads\Export  project - channels,insutaors,conductors - 5_8_2026 (4).ndjson"
OUTPUT_DIR = "dataset_channels"

CLASS_MAP = {
    "tapping_arm": 0,
    "side_arm": 1,
    "v_cross_arm": 2,
    "insulators": 3,
    "street_light": 4,
    "t_rising": 5
}

os.makedirs(f"{OUTPUT_DIR}/labels/train", exist_ok=True)

with open(NDJSON_PATH, "r") as f:
    for idx, line in enumerate(f):
        if not line.strip(): continue
        try:
            item = json.loads(line)
            # Default dimensions for debugging
            w, h = 1000, 1000 
            
            label_lines = []
            for p_id in item.get("projects", {}):
                for label in item["projects"][p_id].get("labels", []):
                    if "annotations" in label and "objects" in label["annotations"]:
                        for obj in label["annotations"]["objects"]:
                            val = obj.get("value")
                            if val in CLASS_MAP:
                                class_id = CLASS_MAP[val]
                                # POLYGON
                                if "polygon" in obj:
                                    coords = [f"{pt['x']/w} {pt['y']/h}" for pt in obj["polygon"]]
                                    label_lines.append(f"{class_id} " + " ".join(coords))
                                # BOX
                                elif "bounding_box" in obj:
                                    box = obj["bounding_box"]
                                    cx, cy = (box["left"] + box["width"]/2)/w, (box["top"] + box["height"]/2)/h
                                    bw, bh = box["width"]/w, box["height"]/h
                                    label_lines.append(f"{class_id} {cx} {cy} {bw} {bh}")

            if label_lines:
                with open(f"{OUTPUT_DIR}/labels/train/img_{idx}.txt", "w") as f_lbl:
                    f_lbl.write("\n".join(label_lines))
                print(f"Record {idx}: Saved {len(label_lines)} labels.")

        except Exception as e:
            print(f"Error {idx}: {e}")

print("Done with label-only run.")
