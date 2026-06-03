import cv2
import sys

def run_test():
    img_path = "uploads/last_ocr_crop.jpg"
    print(f"[DEBUG] Reading {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        print("[DEBUG] Failed to read image")
        return
    h, w = img.shape[:2]
    print(f"[DEBUG] Size: {w}x{h}")
    
    print("\n--- Testing EasyOCR ---")
    try:
        import easyocr
        print("[EasyOCR] Importing successful")
        reader = easyocr.Reader(["en"], gpu=False)
        print("[EasyOCR] Reader initialized successfully")
        
        # Test OCR on full crop
        res = reader.readtext(img, detail=0)
        print(f"Full image raw text: {res}")
        
        # Test OCR on split lines
        from ocr_utils import PoleOCR
        ocr = PoleOCR()
        patch_box = ocr._find_black_patch(img)
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
            extended_crop = img[ey1:ey2, ex1:ex2]
            
            print(f"[EasyOCR] Running on extended crop (size {extended_crop.shape[1]}x{extended_crop.shape[0]}):")
            res_ext = reader.readtext(extended_crop, detail=0)
            print(f"Extended crop raw text: {res_ext}")
            
            band_crop = ocr._find_white_text_band(extended_crop)
            target = band_crop if band_crop is not None else extended_crop
            
            lines = ocr._extract_text_line_crops(target)
            print(f"[EasyOCR] Lines count: {len(lines)}")
            for idx, line in enumerate(lines):
                res_line = reader.readtext(line, detail=0)
                print(f"Line {idx} raw text: {res_line}")
                
            fractional = ocr._fractional_line_crops(target)
            print(f"[EasyOCR] Fractional lines count: {len(fractional)}")
            for idx, line in enumerate(fractional):
                res_line = reader.readtext(line, detail=0)
                print(f"Fractional Line {idx} raw text: {res_line}")
    except Exception as e:
        print(f"[EasyOCR] Failed: {e}")
        
    print("\n--- Testing Pytesseract ---")
    try:
        import pytesseract
        print("[Pytesseract] Importing successful")
        txt = pytesseract.image_to_string(img)
        print(f"Raw Tesseract text: {repr(txt.strip())}")
    except Exception as e:
        print(f"[Pytesseract] Failed: {e}")

if __name__ == "__main__":
    run_test()
