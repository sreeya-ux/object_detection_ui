import cv2
import numpy as np
import pytesseract
from PIL import Image

def _find_black_patch(crop_bgr, pct=0.40):
    try:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        if h < 10 or w < 10: return None
        mean_val = float(np.mean(gray))
        # Use the pct parameter
        thresh_val = min(80, max(20, int(mean_val * pct)))
        print(f"Mean: {mean_val:.2f}, Threshold: {thresh_val}")
        _, dark_mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        min_area = max(150, int(h * w * 0.002))
        best = None
        best_area = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area: continue
            bx, by, bw, bh = cv2.boundingRect(c)
            if bw < 8 or bh < 8: continue
            ar = max(bw, bh) / max(min(bw, bh), 1)
            if ar > 20: continue
            if area > best_area:
                best_area = area
                best = (bx, by, bx + bw, by + bh)
        return best
    except Exception as e:
        print(f"Error: {e}")
        return None

def test_on_file():
    img_path = "/home/ubuntu/object_detection_ui/uploads/ad5df48c-27e5-45de-b304-c0a7be7cb470.jpg"
    image = cv2.imread(img_path)
    if image is None:
        print("Image not found")
        return
        
    h, w = image.shape[:2]
    # Since it's a close-up, use the whole image as pole crop
    patch_box = _find_black_patch(image, pct=0.40)
    if patch_box:
        bx1, by1, bx2, by2 = patch_box
        print(f"Found patch box with pct=0.40: {patch_box}")
        target_crop = image[by1:by2, bx1:bx2]
    else:
        print("No patch box found with pct=0.40")
        target_crop = image
        
    # Resize/upscale for OCR
    min_side = 400
    h_c, w_c = target_crop.shape[:2]
    if max(h_c, w_c) > 0 and max(h_c, w_c) < min_side:
        scale = min_side / max(h_c, w_c)
        target_crop_resized = cv2.resize(target_crop, (int(w_c * scale), int(h_c * scale)), interpolation=cv2.INTER_CUBIC)
    else:
        target_crop_resized = target_crop
        
    cv2.imwrite("/home/ubuntu/object_detection_ui/temp_patch_crop_040.jpg", target_crop_resized)
    
    print("\n--- Testing Tesseract on the crop ---")
    for psm in [3, 4, 6, 7, 8, 11, 12]:
        text = pytesseract.image_to_string(target_crop_resized, config=f'--oem 3 --psm {psm}')
        print(f"PSM {psm}: {repr(text.strip())}")

if __name__ == '__main__':
    test_on_file()
