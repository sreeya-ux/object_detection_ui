import os
import shutil
import zipfile
from math import ceil

# =========================================
# CONFIG
# =========================================

SOURCE_FOLDER = r"D:\split_output\rdss_new"   # CHANGE THIS
OUTPUT_FOLDER = r"D:\split_output"         # CHANGE THIS

# Image extensions
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# =========================================
# CREATE OUTPUT FOLDERS
# =========================================

part1_folder = os.path.join(OUTPUT_FOLDER, "newpart1")
part2_folder = os.path.join(OUTPUT_FOLDER, "newpart2")

os.makedirs(part1_folder, exist_ok=True)
os.makedirs(part2_folder, exist_ok=True)

# =========================================
# GET ALL IMAGES
# =========================================

all_images = [
    f for f in os.listdir(SOURCE_FOLDER)
    if f.lower().endswith(IMAGE_EXTENSIONS)
]

total_images = len(all_images)

if total_images == 0:
    print("No images found.")
    exit()

print(f"\nTOTAL IMAGES FOUND: {total_images}")

# =========================================
# SPLIT INTO 2 PARTS
# =========================================

mid = ceil(total_images / 2)

part1_images = all_images[:mid]
part2_images = all_images[mid:]

print(f"Part 1 Images: {len(part1_images)}")
print(f"Part 2 Images: {len(part2_images)}")

# =========================================
# COPY IMAGES
# =========================================

print("\nCopying Part 1...")
for img in part1_images:
    shutil.copy2(
        os.path.join(SOURCE_FOLDER, img),
        os.path.join(part1_folder, img)
    )

print("Copying Part 2...")
for img in part2_images:
    shutil.copy2(
        os.path.join(SOURCE_FOLDER, img),
        os.path.join(part2_folder, img)
    )

# =========================================
# ZIP FUNCTION
# =========================================

def zip_folder(folder_path, zip_name):
    zip_path = os.path.join(OUTPUT_FOLDER, zip_name)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)

                arcname = os.path.relpath(file_path, folder_path)

                zipf.write(file_path, arcname)

    print(f"Created ZIP: {zip_path}")

# =========================================
# CREATE ZIP FILES
# =========================================

print("\nCreating ZIP files...")

zip_folder(part1_folder, "part1.zip")
zip_folder(part2_folder, "part2.zip")

print("\nDONE ✅")