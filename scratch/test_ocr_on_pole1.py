import cv2
import base64
import requests
import json
import numpy as np
from PIL import Image
from io import BytesIO
from ocr_utils import PoleOCR
from config import AZURE_MISTRAL_API_KEY, AZURE_MISTRAL_ENDPOINT, AZURE_MISTRAL_MODEL

def run_test():
    img_path = "uploads/0748854a-f0dd-4be5-98b4-3c6a54545831.jpg"
    img = cv2.imread(img_path)
    if img is None:
        print("Image not found")
        return
        
    ocr = PoleOCR()
    
    # Pole 1 box: (4, 794, 214, 1439)
    x1, y1, x2, y2 = 4, 794, 214, 1439
    
    # Let's run process_pole_tag on Pole 1 manually and see what it does
    print("\n--- Running process_pole_tag on Pole 1 box (4, 794, 214, 1439) ---")
    res = ocr.process_pole_tag(img, [x1, y1, x2, y2])
    print(f"process_pole_tag result: {repr(res)}")
    
    # Let's inspect the crop paddings and sub-crops for Pole 1
    h_orig, w_orig = img.shape[:2]
    box_w = x2 - x1
    box_h = y2 - y1
    
    pad_left = max(20, int(box_w * 1.8))
    pad_right = max(20, int(box_w * 0.45))
    pad_top = max(20, int(box_h * 0.06))
    pad_bottom = max(20, int(box_h * 0.08))
    
    px1 = max(0, x1 - pad_left)
    py1 = max(0, y1 - pad_top)
    px2 = min(w_orig, x2 + pad_right)
    py2 = min(h_orig, y2 + pad_bottom)
    
    pole_crop = img[py1:py2, px1:px2]
    cv2.imwrite("scratch/diagnose_pole1_crop.jpg", pole_crop)
    
    patch_box = ocr._find_black_patch(pole_crop)
    print(f"Patch box inside Pole 1: {patch_box}")
    
    if patch_box:
        bx1, by1, bx2, by2 = patch_box
        ph, pw = pole_crop.shape[:2]
        pad_x = max(18, pw // 20)
        pad_top = max(18, ph // 20)
        pad_bottom = max(40, ph // 8)
        ex1 = max(0, bx1 - pad_x)
        ey1 = max(0, by1 - pad_top)
        ex2 = min(pw, bx2 + pad_x)
        ey2 = min(ph, by2 + pad_bottom)
        
        extended_crop = pole_crop[ey1:ey2, ex1:ex2]
        cv2.imwrite("scratch/diagnose_pole1_extended_crop.jpg", extended_crop)
        print(f"Saved extended crop (size {extended_crop.shape}) to scratch/diagnose_pole1_extended_crop.jpg")
        
        # Test Mistral OCR on the extended crop
        print("\n--- Running Azure Mistral on Pole 1 Extended Crop ---")
        crop_rgb = cv2.cvtColor(extended_crop, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(crop_rgb)
        buf = BytesIO()
        pil.save(buf, format="JPEG", quality=92)
        b64_data = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {AZURE_MISTRAL_API_KEY}"
        }
        payload = {
            "model": AZURE_MISTRAL_MODEL,
            "document": {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{b64_data}"
            },
            "include_image_base64": False
        }
        
        r = requests.post(AZURE_MISTRAL_ENDPOINT, headers=headers, json=payload, timeout=10)
        if r.status_code == 200:
            res_json = r.json()
            pages = res_json.get("pages", [])
            if pages:
                markdown = pages[0].get("markdown", "")
                print(f"Raw Mistral response: {repr(markdown)}")
                print(f"Normalized ID: {repr(ocr._normalize_pole_id(markdown))}")
            else:
                print("Empty pages list")
        else:
            print(f"Failed with status: {r.status_code}: {r.text}")
            
        # Test EasyOCR
        try:
            import easyocr
            reader = easyocr.Reader(["en"], gpu=False)
            res_easy = reader.readtext(extended_crop, detail=0)
            print(f"EasyOCR response: {res_easy}")
        except Exception as e:
            print(f"EasyOCR failed: {e}")

if __name__ == "__main__":
    run_test()
