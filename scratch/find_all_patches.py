import cv2
import numpy as np

def run_test():
    img_path = "scratch/diagnose_pole_crop.jpg"
    crop_bgr = cv2.imread(img_path)
    if crop_bgr is None:
        print("Error: Could not read scratch/diagnose_pole_crop.jpg")
        return
        
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    print(f"Pole crop shape: {w}x{h}")
    
    mean_val = float(np.mean(gray))
    thresh_val = min(80, max(20, int(mean_val * 0.55)))
    print(f"Mean gray value: {mean_val:.2f}, Threshold value used: {thresh_val}")
    
    _, dark_mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dark_mask_closed = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dark_mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Total contours found: {len(contours)}")
    
    min_area = max(150, int(h * w * 0.002))
    print(f"Min area threshold: {min_area}")
    
    candidates = []
    for idx, c in enumerate(contours):
        area = cv2.contourArea(c)
        bx, by, bw, bh = cv2.boundingRect(c)
        ar = max(bw, bh) / max(min(bw, bh), 1)
        
        is_valid = True
        reason = "OK"
        if area < min_area:
            is_valid = False
            reason = f"area {area:.1f} < min_area {min_area}"
        elif bw < 8 or bh < 8:
            is_valid = False
            reason = f"dimensions too small ({bw}x{bh})"
        elif ar > 20:
            is_valid = False
            reason = f"aspect ratio {ar:.1f} > 20"
            
        candidates.append({
            "idx": idx,
            "area": area,
            "box": (bx, by, bx+bw, by+bh),
            "width": bw,
            "height": bh,
            "ar": ar,
            "is_valid": is_valid,
            "reason": reason
        })
        
    # Sort candidates by area descending
    candidates.sort(key=lambda x: x["area"], reverse=True)
    
    print("\n--- Detected Contours (Sorted by Area) ---")
    for cand in candidates[:15]:
        status = "VALID" if cand["is_valid"] else "INVALID"
        print(f"Contour {cand['idx']}: Status={status}, Area={cand['area']:.1f}, Box={cand['box']}, Size={cand['width']}x{cand['height']}, AR={cand['ar']:.2f}, Reason={cand['reason']}")

if __name__ == "__main__":
    run_test()
