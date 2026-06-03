import cv2
import easyocr

def run_test():
    img_path = "uploads/0748854a-f0dd-4be5-98b4-3c6a54545831.jpg"
    img = cv2.imread(img_path)
    if img is None:
        print("Image not found")
        return
        
    print(f"Image shape: {img.shape}")
    
    print("Initializing EasyOCR...")
    reader = easyocr.Reader(["en"], gpu=False)
    
    print("Running EasyOCR on full image...")
    res = reader.readtext(img, detail=1)
    
    print(f"\n--- EasyOCR Results (Total: {len(res)}) ---")
    for idx, (bbox, text, conf) in enumerate(res):
        print(f"Result {idx}: '{text}' (conf: {conf:.2f}) at bbox: {bbox}")
        
    # Also test pytesseract on full image
    print("\nRunning Pytesseract on full image...")
    try:
        import pytesseract
        txt = pytesseract.image_to_string(img)
        print("Pytesseract text:")
        print(repr(txt))
    except Exception as e:
        print(f"Pytesseract failed: {e}")

if __name__ == "__main__":
    run_test()
