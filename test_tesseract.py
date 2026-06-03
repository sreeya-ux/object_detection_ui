import cv2
import pytesseract
import numpy as np
from PIL import Image

def test_ocr(img_path):
    print(f"=== Testing {img_path} ===")
    img = cv2.imread(img_path)
    if img is None:
        print("Failed to read image")
        return
        
    # Let's try raw OCR on full image first
    text_raw = pytesseract.image_to_string(img, config='--oem 3 --psm 11')
    print("--- RAW FULL IMAGE ---")
    print(repr(text_raw.strip()))
    
    # Try grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Try simple thresholding and inversion (standard white-on-black to black-on-white)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    text_thresh = pytesseract.image_to_string(thresh, config='--oem 3 --psm 11')
    print("--- THRESHOLD INV ---")
    print(repr(text_thresh.strip()))
    
    # Try Otsu thresholding and inversion
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    text_otsu = pytesseract.image_to_string(otsu, config='--oem 3 --psm 11')
    print("--- OTSU INV ---")
    print(repr(text_otsu.strip()))
    
    # Try adaptive thresholding
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    text_adaptive = pytesseract.image_to_string(adaptive, config='--oem 3 --psm 11')
    print("--- ADAPTIVE INV ---")
    print(repr(text_adaptive.strip()))

if __name__ == '__main__':
    import glob
    files = glob.glob("temp_*.jpg")
    for f in files:
        test_ocr(f)
