import cv2
import numpy as np
import pytesseract
from PIL import Image

def test_ocr_30():
    img_path = "/home/ubuntu/object_detection_ui/uploads/ad5df48c-27e5-45de-b304-c0a7be7cb470.jpg"
    image = cv2.imread(img_path)
    if image is None:
        print("Image not found")
        return
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Locate black patch with threshold = 30
    _, dark_mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best = None
    if contours:
        h, w = gray.shape[:2]
        min_area = max(150, int(h * w * 0.002))
        best_area = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area: continue
            bx, by, bw, bh = cv2.boundingRect(c)
            ar = max(bw, bh) / max(min(bw, bh), 1)
            if ar > 20: continue
            if area > best_area:
                best_area = area
                best = (bx, by, bx + bw, by + bh)
                
    if best:
        bx1, by1, bx2, by2 = best
        print(f"Crop box: {best}")
        patch = gray[by1:by2, bx1:bx2]
    else:
        print("No patch found, using full image")
        patch = gray

    # Invert
    inverted = cv2.bitwise_not(patch)
    
    # Save crop
    cv2.imwrite("/home/ubuntu/object_detection_ui/temp_inverted_patch_30.jpg", inverted)
    
    # Upscale
    min_side = 400
    h_c, w_c = inverted.shape[:2]
    if max(h_c, w_c) > 0 and max(h_c, w_c) < min_side:
        scale = min_side / max(h_c, w_c)
        inverted_resized = cv2.resize(inverted, (int(w_c * scale), int(h_c * scale)), interpolation=cv2.INTER_CUBIC)
    else:
        inverted_resized = inverted
        
    print("\n--- Running Tesseract OCR on Inverted Grayscale Crop (Threshold 30) ---")
    for psm in [3, 4, 6, 7, 8, 11, 12]:
        text = pytesseract.image_to_string(inverted_resized, config=f'--oem 3 --psm {psm}')
        print(f"PSM {psm}: {repr(text.strip())}")

if __name__ == '__main__':
    test_ocr_30()
