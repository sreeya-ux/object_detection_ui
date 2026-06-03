import cv2
import pytesseract
from PIL import Image
import numpy as np

def _find_black_patch(crop_bgr):
    try:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        if h < 10 or w < 10: return None
        mean_val = float(np.mean(gray))
        thresh_val = min(80, max(20, int(mean_val * 0.55)))
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

def _enhance_for_ocr(crop_bgr):
    try:
        lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) # Grayscale
        # Let's try simple thresholding for Tesseract instead of heavy sharpening
        return crop_bgr
    except Exception:
        return crop_bgr

def test_on_file():
    img_path = "uploads/6815587f-7631-4587-b510-13e5be4df393.jpg"
    image = cv2.imread(img_path)
    if image is None:
        print("Image not found")
        return
        
    # Mocking pole box
    h_orig, w_orig = image.shape[:2]
    # For this image, the pole is probably the full image or a large part of it.
    # Let's try using the whole image as pole box
    box = [0, 0, w_orig, h_orig]
    
    x1, y1, x2, y2 = [int(v) for v in box]
    pole_crop = image[y1:y2, x1:x2]
    
    # Try to find black patch
    patch_box = _find_black_patch(pole_crop)
    if patch_box:
        bx1, by1, bx2, by2 = patch_box
        print(f"Found patch box: {patch_box}")
        target_crop = pole_crop[by1:by2, bx1:bx2]
    else:
        print("No patch box found")
        target_crop = pole_crop
        
    # Let's test Tesseract on target_crop raw vs enhanced
    print("\n--- Testing Raw target_crop ---")
    for psm in [3, 4, 6, 11, 12]:
        text = pytesseract.image_to_string(target_crop, config=f'--oem 3 --psm {psm}')
        print(f"PSM {psm}: {repr(text.strip())}")
        
    # Let's test with simple resize/upscale
    print("\n--- Testing Upscaled target_crop ---")
    h_c, w_c = target_crop.shape[:2]
    upscaled = cv2.resize(target_crop, (w_c*2, h_c*2), interpolation=cv2.INTER_CUBIC)
    for psm in [3, 6, 11]:
        text = pytesseract.image_to_string(upscaled, config=f'--oem 3 --psm {psm}')
        print(f"PSM {psm}: {repr(text.strip())}")

if __name__ == '__main__':
    test_on_file()
