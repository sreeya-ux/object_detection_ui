import cv2
import numpy as np

def test_fixed(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    print(f"=== Fixed Threshold Tests on {img_path} ===")
    
    for th in [25, 30, 35, 40, 45, 50, 60]:
        _, dark_mask = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        min_area = max(150, int(h * w * 0.002))
        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            ar = max(bw, bh) / max(min(bw, bh), 1)
            if ar > 20:
                continue
            candidates.append((area, (bx, by, bw, bh), ar))
            
        candidates.sort(key=lambda x: x[0], reverse=True)
        print(f"Threshold {th}:")
        if candidates:
            for idx, cand in enumerate(candidates[:2]):
                print(f"  Cand {idx}: area={cand[0]:.1f}, box={cand[1]}, ar={cand[2]:.2f}")
        else:
            print("  No candidates")

test_fixed("image2_remote.jpg")
