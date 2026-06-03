import cv2
from ocr_utils import PoleOCR

def run_test():
    img_path = "uploads/6815587f-7631-4587-b510-13e5be4df393_patch.jpg"
    print(f"[DEBUG] Reading {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        print("[DEBUG] Failed to read image")
        return
    h, w = img.shape[:2]
    print(f"[DEBUG] Size: {w}x{h}")
    
    ocr = PoleOCR()
    
    # Run _request_azure_text on the full patch
    print("\n--- Running Azure Mistral raw text on full patch ---")
    raw_text = ocr._request_azure_text(img)
    print(f"Raw text: {repr(raw_text)}")
    
    # Run candidate crops
    # Let's inspect unique lines and candidates
    patch_box = ocr._find_black_patch(img)
    print(f"Patch box inside patch: {patch_box}")
    
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
    
    # Test line crops
    band_crop = ocr._find_white_text_band(target)
    line_target = band_crop if band_crop is not None else target
    if band_crop is not None:
        print(f"Band crop size: {band_crop.shape[1]}x{band_crop.shape[0]}")
    
    lines = ocr._extract_text_line_crops(line_target)
    print(f"Lines count: {len(lines)}")
    for idx, line in enumerate(lines):
        raw_line = ocr._request_azure_text(line)
        print(f"Line {idx} raw text: {repr(raw_line)}")
        
    fractional = ocr._fractional_line_crops(line_target)
    print(f"Fractional count: {len(fractional)}")
    for idx, line in enumerate(fractional):
        raw_line = ocr._request_azure_text(line)
        print(f"Fractional Line {idx} raw text: {repr(raw_line)}")
        
    print("\n--- Running process_pole_tag ---")
    res = ocr.process_pole_tag(img, [0, 0, w, h])
    print(f"\nResult: {res}\n")

if __name__ == "__main__":
    run_test()
