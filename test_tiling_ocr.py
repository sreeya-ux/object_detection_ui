
import cv2
from ocr_utils import PoleOCR

def test_tiling_result(image_path):
    print(f"Starting Vertical Tiling OCR on {image_path}...")
    ocr = PoleOCR()
    img = cv2.imread(image_path)
    if img is None:
        return

    h, w = img.shape[:2]
    # We will 'simulate' a perfect pole detection: a thin strip in the center
    # This proves the OCR works if the detection is good
    x1, x2 = int(w*0.35), int(w*0.50) # The pole is roughly here
    y1, y2 = int(h*0.1), int(h*0.9)
    pole_box = [x1, y1, x2, y2]
    
    result = ocr.process_pole_tag(img, pole_box)
    print("\n--- TILING OCR RESULTS ---")
    print(f"POLE ID: {result}")
    print("--------------------------")

if __name__ == "__main__":
    test_tiling_result(r"c:\Users\ASK037-PC\.gemini\antigravity\brain\8b4adf17-a991-49b2-ae58-0a654dd7c2df\media__1778150808440.jpg")
