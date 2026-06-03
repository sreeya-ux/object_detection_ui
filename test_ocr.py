"""
Quick OCR test - run on any image file to see what the OCR reads.
Usage:  venv\\Scripts\\python.exe test_ocr.py <image_path>
"""
import sys
import cv2
import numpy as np
from ocr_utils import PoleOCR

def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else "result.jpg"
    print(f"Testing OCR on: {img_path}")
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"ERROR: Could not read {img_path}")
        return

    h, w = img.shape[:2]
    print(f"Image size: {w}x{h}")

    # Use the full image as the "pole box" so we test the whole thing
    box = [0, 0, w, h]

    ocr = PoleOCR()
    if not ocr.active:
        print("OCR not active - check local EasyOCR installation")
        return

    # Also test patch detection
    patch_box = ocr._find_black_patch(img)
    print(f"Black patch detected: {patch_box}")

    result = ocr.process_pole_tag(img, box)
    print(f"\n=== OCR RESULT: {result} ===\n")

if __name__ == "__main__":
    main()
