import cv2
import os
import glob
from ocr_utils import PoleOCR

def test_all():
    ocr = PoleOCR()
    files = glob.glob("uploads/*.jpg")
    print(f"Found {len(files)} files in uploads/")
    for f in sorted(files):
        if "_patch" in f or "last_ocr" in f:
            continue
        print(f"\n======================================")
        print(f"Testing file: {f}")
        img = cv2.imread(f)
        if img is None:
            continue
        h, w = img.shape[:2]
        
        # Test full image OCR
        print("--- Running process_pole_tag (full image) ---")
        res = ocr.process_pole_tag(img, [0, 0, w, h])
        print(f"Result: {res}")

if __name__ == "__main__":
    test_all()
