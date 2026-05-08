
import easyocr
import cv2
import numpy as np

def test_vertical_paragraphs(image_path):
    print(f"Starting Vertical Paragraph Scan on {image_path}...")
    reader = easyocr.Reader(['en'], gpu=False)
    
    img = cv2.imread(image_path)
    if img is None:
        return

    # No rotation! Keep it upright.
    # We use 'paragraph=True' to help it group the stacked letters
    results = reader.readtext(img, paragraph=True, detail=1)

    print("\n--- VERTICAL PARAGRAPH RESULTS ---")
    for (bbox, text) in results:
        # EasyOCR in paragraph mode returns (bbox, text)
        print(f"  - '{text}'")

if __name__ == "__main__":
    test_vertical_paragraphs(r"c:\Users\ASK037-PC\.gemini\antigravity\brain\8b4adf17-a991-49b2-ae58-0a654dd7c2df\media__1778150808440.jpg")
