
import cv2
from ocr_utils import PoleOCR

def test_full_angled_scan(image_path):
    print(f"Starting Full-Image Tiling Scan (Angled Mode) on {image_path}...")
    ocr = PoleOCR()
    img = cv2.imread(image_path)
    if img is None:
        return

    h, w = img.shape[:2]
    # We force a 'fake' box that covers the entire middle area of the image
    # To search for text everywhere
    fake_box = [int(w*0.1), int(h*0.1), int(w*0.9), int(h*0.9)]
    
    result = ocr.process_pole_tag(img, fake_box)
    print("\n--- FULL IMAGE OCR RESULTS ---")
    print(f"POLE ID FOUND: {result}")
    print("-------------------------------")

if __name__ == "__main__":
    test_full_angled_scan(r"c:\Users\ASK037-PC\.gemini\antigravity\brain\8b4adf17-a991-49b2-ae58-0a654dd7c2df\media__1778153905670.jpg")
