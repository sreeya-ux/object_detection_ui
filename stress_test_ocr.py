
import easyocr
import cv2
import numpy as np

def stress_test_ocr(image_path):
    print(f"STRESS TEST: Loading {image_path}...")
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image.")
        return
    
    reader = easyocr.Reader(['en'], gpu=False)
    
    print("STRESS TEST: Reading full image (high-res mode)...")
    results = reader.readtext(image_path, detail=1)
    
    print("\n--- RESULTS ---")
    if not results:
        print("No text found in full image.")
    else:
        for (bbox, text, prob) in results:
            print(f"Found: '{text}' (conf: {prob:.2f}) at {bbox}")

if __name__ == "__main__":
    stress_test_ocr(r"c:\Users\ASK037-PC\.gemini\antigravity\brain\8b4adf17-a991-49b2-ae58-0a654dd7c2df\media__1778150808440.jpg")
