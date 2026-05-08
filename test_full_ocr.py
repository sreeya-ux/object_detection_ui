
import easyocr
import cv2
import numpy as np

def test_full_image(image_path):
    print(f"Starting Full Image OCR Scan on {image_path}...")
    reader = easyocr.Reader(['en'], gpu=False)
    
    img = cv2.imread(image_path)
    if img is None:
        print("Could not open image.")
        return

    # 1. Scan Original (Horizontal)
    print("Scanning Horizontal pass...")
    results_h = reader.readtext(img)
    
    # 2. Scan Rotated (Vertical)
    print("Scanning Vertical pass (rotating 90 deg)...")
    img_rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    results_v = reader.readtext(img_rot)

    print("\n--- OCR RESULTS FOUND ---")
    all_text = []
    
    print("\n[Horizontal Findings]:")
    for (bbox, text, conf) in results_h:
        if conf > 0.2:
            print(f"  - {text} (conf: {conf:.2f})")
            all_text.append(text)

    print("\n[Vertical Findings (Pole Text)]: ")
    for (bbox, text, conf) in results_v:
        if conf > 0.2:
            print(f"  - {text} (conf: {conf:.2f})")
            all_text.append(text)

    print("\n-------------------------")
    print(f"FINAL COMBINED ID: {' '.join(all_text)}")

if __name__ == "__main__":
    test_full_image(r"c:\Users\ASK037-PC\.gemini\antigravity\brain\8b4adf17-a991-49b2-ae58-0a654dd7c2df\media__1778150808440.jpg")
