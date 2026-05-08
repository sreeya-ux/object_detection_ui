
import easyocr
import cv2
import numpy as np

def test_ultra_scan(image_path):
    print(f"Starting Ultra OCR Scan on {image_path}...")
    reader = easyocr.Reader(['en'], gpu=False)
    
    img = cv2.imread(image_path)
    if img is None:
        return

    # 1. Resize for better detail (2x)
    img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    
    # 2. Preprocess
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # 3. Rotate for Vertical Pass
    img_rot = cv2.rotate(enhanced, cv2.ROTATE_90_CLOCKWISE)

    print("Searching Vertical pass with enhancement...")
    results_v = reader.readtext(img_rot, detail=1)

    print("\n--- ULTRA SCAN RESULTS ---")
    for (bbox, text, conf) in results_v:
        # Print EVERYTHING found
        print(f"  - '{text}' (conf: {conf:.2f})")

if __name__ == "__main__":
    test_ultra_scan(r"c:\Users\ASK037-PC\.gemini\antigravity\brain\8b4adf17-a991-49b2-ae58-0a654dd7c2df\media__1778150808440.jpg")
