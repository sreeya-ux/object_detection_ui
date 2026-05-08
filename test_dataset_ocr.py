
import cv2
from ocr_utils import PoleOCR

def test_dataset_ocr(image_path):
    print(f"Starting Dataset OCR Scan on {image_path}...")
    ocr = PoleOCR()
    img = cv2.imread(image_path)
    if img is None:
        print("Could not find image.")
        return

    h, w = img.shape[:2]
    # In this dataset, poles are usually in the middle
    fake_box = [int(w*0.3), int(h*0.1), int(w*0.7), int(h*0.9)]
    
    result = ocr.process_pole_tag(img, fake_box)
    print("\n--- DATASET OCR RESULTS ---")
    print(f"POLE ID FOUND: {result}")
    print("----------------------------")

if __name__ == "__main__":
    import os
    img_path = r"training_data\images\val\ds1_img_137.jpg"
    if os.path.exists(img_path):
        test_dataset_ocr(img_path)
    else:
        print(f"File not found: {img_path}")
