import os
import shutil
from pathlib import Path

# INPUT FOLDER
input_folder = r"C:\Users\asus\Downloads\rdss_gis_final"

# OUTPUT FOLDER
output_folder = r"D:\split_output\rdss_new"

# Create output folder
os.makedirs(output_folder, exist_ok=True)

# Supported image extensions
image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

# Get all images
images = [
    f for f in os.listdir(input_folder)
    if Path(f).suffix.lower() in image_exts
]

# Sort images (important for consistent numbering)
images.sort()

# Rename with incremental numbers
for idx, image_name in enumerate(images, start=1):

    ext = Path(image_name).suffix.lower()

    # Example: img_000001.jpg
    new_name = f"img_{idx:06d}{ext}"

    src = os.path.join(input_folder, image_name)
    dst = os.path.join(output_folder, new_name)

    shutil.copy2(src, dst)

    print(f"{image_name}  -->  {new_name}")

print("\nDone bro ✅")
print(f"Renamed images saved in:\n{output_folder}")