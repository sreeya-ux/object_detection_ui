
import cv2
from ocr_utils import PoleOCR

def test_dataset_ocr_specific(image_path):
    print(f"Starting OCR Scan on {image_path}...")
    ocr = PoleOCR()
    img = cv2.imread(image_path)
    if img is None:
        return

    h, w = img.shape[:2]
    # The text is at the bottom center
    fake_box = [int(w*0.45), int(h*0.85), int(w*0.55), int(h*0.99)]
    
    result = ocr.process_pole_tag(img, fake_box)
    print("\n--- OCR RESULT ---")
    print(f"POLE ID: {result}")
    print("-------------------")

if __name__ == "__main__":
    test_dataset_ocr_specific(r"training_data\images\val\ds1_img_454.jpg")
