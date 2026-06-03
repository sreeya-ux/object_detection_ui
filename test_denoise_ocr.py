import cv2
import numpy as np
import pytesseract

def test_denoise():
    img_path = "/home/ubuntu/object_detection_ui/temp_inverted_patch_30.jpg"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Image not found")
        return
        
    for ksize in [3, 5, 7, 9]:
        # Apply median blur to remove specks
        blurred = cv2.medianBlur(img, ksize)
        
        # Binarize using Otsu
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Add white border
        margin = 30
        clean_img = cv2.copyMakeBorder(binary, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=255)
        
        # Save crop
        cv2.imwrite(f"/home/ubuntu/object_detection_ui/temp_denoise_{ksize}.jpg", clean_img)
        
        # Run Tesseract
        print(f"\n--- Median Blur ksize={ksize} + Otsu ---")
        config_whitelist = '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/- '
        for psm in [3, 6, 11, 12]:
            text = pytesseract.image_to_string(clean_img, config=f'--oem 3 --psm {psm}')
            text_w = pytesseract.image_to_string(clean_img, config=f'--oem 3 --psm {psm} {config_whitelist}')
            print(f"  PSM {psm} (Normal): {repr(text.strip())}")
            print(f"  PSM {psm} (Whitelist): {repr(text_w.strip())}")

if __name__ == '__main__':
    test_denoise()
