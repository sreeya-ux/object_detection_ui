import cv2
from ocr_utils import PoleOCR

def run_test():
    img_path = "uploads/0748854a-f0dd-4be5-98b4-3c6a54545831.jpg"
    print(f"[DEBUG] Reading {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        print("[DEBUG] Failed to read image")
        return
    h, w = img.shape[:2]
    print(f"[DEBUG] Size: {w}x{h}")
    
    ocr = PoleOCR()
    
    # 1. Run Azure Mistral raw text on full image
    print("\n--- Running Azure Mistral raw text on full image ---")
    raw_text = ocr._request_azure_text(img)
    print(f"Raw text: {repr(raw_text)}")
    
    # 2. Find black patch
    patch_box = ocr._find_black_patch(img)
    print(f"Black patch: {patch_box}")
    
    target = img
    if patch_box:
        bx1, by1, bx2, by2 = patch_box
        ph, pw = img.shape[:2]
        pad_x = max(18, pw // 20)
        pad_top = max(18, ph // 20)
        pad_bottom = max(40, ph // 8)
        ex1 = max(0, bx1 - pad_x)
        ey1 = max(0, by1 - pad_top)
        ex2 = min(pw, bx2 + pad_x)
        ey2 = min(ph, by2 + pad_bottom)
        target = img[ey1:ey2, ex1:ex2]
        print(f"Extended crop size: {target.shape[1]}x{target.shape[0]}")
        
    print("\n--- Running Azure Mistral on target crop ---")
    raw_target = ocr._request_azure_text(target)
    print(f"Raw target: {repr(raw_target)}")
    
    # 3. Running EasyOCR
    try:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=False)
        print("\n--- Running EasyOCR on target ---")
        res_ext = reader.readtext(target, detail=0)
        print(f"EasyOCR raw text: {res_ext}")
    except Exception as e:
        print(f"EasyOCR failed: {e}")
        
    # 4. Running Tesseract
    try:
        import pytesseract
        print("\n--- Running Pytesseract on target ---")
        txt = pytesseract.image_to_string(target)
        print(f"Tesseract raw text: {repr(txt.strip())}")
    except Exception as e:
        print(f"Tesseract failed: {e}")

if __name__ == "__main__":
    run_test()
