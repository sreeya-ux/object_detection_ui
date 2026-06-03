import cv2
import numpy as np
import pytesseract

def test_binarize():
    img_path = "/home/ubuntu/object_detection_ui/temp_inverted_patch_30.jpg"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Image not found")
        return
        
    # We want dark text on white background.
    # The image is already inverted, so text is dark (low values) and bg is light (high values).
    # Let's apply thresholding to make bg pure white (255) and text pure black (0).
    # We can try Otsu and simple threshold values.
    
    # Try simple threshold at 100, 120, 140
    for thresh_val in [100, 120, 130, 140, 150]:
        _, thresh = cv2.threshold(img, thresh_val, 255, cv2.THRESH_BINARY)
        # Save thresh image
        cv2.imwrite(f"/home/ubuntu/object_detection_ui/temp_bin_{thresh_val}.jpg", thresh)
        
        # Test OCR with and without whitelist
        config_normal = '--oem 3 --psm 6'
        config_whitelist = '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/- '
        
        text_norm = pytesseract.image_to_string(thresh, config=config_normal)
        text_white = pytesseract.image_to_string(thresh, config=config_whitelist)
        
        print(f"\nThreshold {thresh_val}:")
        print(f"  Normal: {repr(text_norm.strip())}")
        print(f"  Whitelist: {repr(text_white.strip())}")
        
        # Test PSM 3 and 11
        for psm in [3, 11, 12]:
            t = pytesseract.image_to_string(thresh, config=f'--oem 3 --psm {psm}')
            print(f"  PSM {psm}: {repr(t.strip())}")

if __name__ == '__main__':
    test_binarize()
