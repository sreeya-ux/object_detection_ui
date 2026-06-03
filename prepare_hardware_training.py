import os
import shutil
from pathlib import Path

# Config
SRC2 = Path(r"dataset 2 labels&images/dataset 2")
DEST = Path(r"training_data_hardware")

# Target Classes for Hardware Model
# 0: insulator, 1: crossarm, 2: conductor
CLASSES = ["insulator", "crossarm", "conductor"]

# Mapping for Dataset 2
# Usually: 0: channel, 1: insulator, 2: conductor
MAP2 = {
    "0": "1", # channel -> crossarm
    "1": "0", # insulator -> insulator
    "2": "2", # conductor -> conductor
}

def prepare():
    os.makedirs(DEST / "images/train", exist_ok=True)
    os.makedirs(DEST / "labels/train", exist_ok=True)
    
    total = 0
    
    # Process DS2 Only
    print("Processing Dataset 2 (Hardware Only)...")
    for split in ["train", "val"]:
        img_dir = SRC2 / "images" / split
        lbl_dir = SRC2 / "labels" / split
        if not img_dir.exists(): continue
        
        for img_file in img_dir.iterdir():
            lbl_file = lbl_dir / (img_file.stem + ".txt")
            if lbl_file.exists():
                shutil.copy2(img_file, DEST / "images/train" / f"hw_{img_file.name}")
                with open(lbl_file, "r") as f:
                    lines = f.readlines()
                new_lines = []
                for line in lines:
                    parts = line.split()
                    if parts and parts[0] in MAP2:
                        parts[0] = MAP2[parts[0]]
                        new_lines.append(" ".join(parts) + "\n")
                with open(DEST / "labels/train" / f"hw_{lbl_file.name}", "w") as f:
                    f.writelines(new_lines)
                total += 1

    # Write data.yaml
    with open(DEST / "data.yaml", "w") as f:
        f.write(f"path: {DEST.absolute()}\n")
        f.write("train: images/train\n")
        f.write("val: images/train\n")
        f.write(f"nc: {len(CLASSES)}\n")
        f.write(f"names: {CLASSES}\n")

    print(f"Done! Prepared {total} images for hardware training in {DEST}")

if __name__ == "__main__":
    prepare()
