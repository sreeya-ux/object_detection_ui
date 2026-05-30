import os
import shutil

folder1 = r"C:\Users\asus\Downloads\rdss_gis_images_old\rdss_gis_images"

folder2 = r"C:\Users\asus\Downloads\rdss_gis_images_new\rdss_gis_images"

output_folder = r"C:\Users\asus\Downloads\rdss_gis_final"

# CREATE OUTPUT FOLDER
os.makedirs(output_folder, exist_ok=True)

# filenames from folder1
folder1_files = set(os.listdir(folder1))

copied = 0
skipped = 0

# process folder2
for file_name in os.listdir(folder2):

    source_path = os.path.join(folder2, file_name)

    # skip folders
    if not os.path.isfile(source_path):
        continue

    # skip duplicates
    if file_name in folder1_files:
        skipped += 1
        print(f"Skipped: {file_name}")
        continue

    try:

        destination = os.path.join(output_folder, file_name)

        shutil.copy2(source_path, destination)

        copied += 1

        print(f"Copied: {file_name}")

    except Exception as e:

        print(f"Error copying {file_name}: {e}")

print("\nDone ✅")
print(f"Copied unique files : {copied}")
print(f"Skipped duplicates  : {skipped}")