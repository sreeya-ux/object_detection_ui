import cv2
import numpy as np

def analyze_patch(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("Error: Could not read image")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    print(f"Image size: {w}x{h}")
    
    mean_val = float(np.mean(gray))
    print(f"Mean grayscale value: {mean_val:.2f}")
    
    # Try different thresholds
    for pct in [0.4, 0.5, 0.55, 0.6, 0.7]:
        thresh_val = min(80, max(20, int(mean_val * pct)))
        _, dark_mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"\n--- Testing threshold pct {pct:.2f} (val: {thresh_val}) ---")
        min_area = max(150, int(h * w * 0.002))
        print(f"Min area: {min_area}")
        
        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            if bw < 8 or bh < 8:
                continue
            ar = max(bw, bh) / max(min(bw, bh), 1)
            if ar > 20:
                continue
            candidates.append((area, (bx, by, bw, bh), ar))
            
        candidates.sort(key=lambda x: x[0], reverse=True)
        for i, cand in enumerate(candidates[:5]):
            area, box, ar = cand
            bx, by, bw, bh = box
            print(f"Candidate {i}: area={area:.1f}, box={box}, aspect_ratio={ar:.2f}, relative_area={area/(w*h):.4f}")

analyze_patch("image2_remote.jpg")
