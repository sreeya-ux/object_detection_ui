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
    img_path = "uploads/6815587f-7631-4587-b510-13e5be4df393.jpg"
    img = cv2.imread(img_path)
    if img is None:
        print("Image not found")
        return
        
    h_orig, w_orig = img.shape[:2]
    print(f"Original image shape: {w_orig}x{h_orig}")
    
    ocr = PoleOCR()
    
    # We simulate a pole detection box at: (4, 794, 214, 1439) or similar (since it's Pole 1)
    # Let's run process_pole_tag with Pole 1's box
    print("\n--- Running process_pole_tag on Pole 1 box ---")
    res = ocr.process_pole_tag(img, [4, 794, 214, 1439])
    print(f"process_pole_tag result: {repr(res)}")
    
    # Let's manually run the candidate crop analysis and print everything
    box = [4, 794, 214, 1439]
    x1, y1, x2, y2 = [int(v) for v in box]
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    
    pad_left = max(20, int(box_w * 1.8))
    pad_right = max(20, int(box_w * 0.45))
    pad_top = max(20, int(box_h * 0.06))
    pad_bottom = max(20, int(box_h * 0.08))
    
    px1 = max(0, x1 - pad_left)
    py1 = max(0, y1 - pad_top)
    px2 = min(w_orig, x2 + pad_right)
    py2 = min(h_orig, y2 + pad_bottom)
    pole_crop = img[py1:py2, px1:px2]
    
    print(f"Pole crop size: {pole_crop.shape[1]}x{pole_crop.shape[0]}")
    
    candidate_crops = []
    line_candidates = []
    
    patch_box = ocr._find_black_patch(pole_crop)
    print(f"Patch box: {patch_box}")
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
        if extended_crop.size > 0:
            candidate_crops.append(extended_crop)
            band_crop = ocr._find_white_text_band(extended_crop)
            if band_crop is not None and band_crop.size > 0:
                candidate_crops.append(band_crop)
                line_candidates.extend(ocr._extract_text_line_crops(band_crop))
                line_candidates.extend(ocr._fractional_line_crops(band_crop))
            else:
                line_candidates.extend(ocr._extract_text_line_crops(extended_crop))
                line_candidates.extend(ocr._fractional_line_crops(extended_crop))
                
    if not candidate_crops:
        candidate_crops.append(pole_crop)
    elif len(candidate_crops) < 2:
        fallback_band = ocr._find_white_text_band(candidate_crops[0])
        candidate_crops.append(fallback_band if fallback_band is not None and fallback_band.size > 0 else pole_crop)
        
    unique_candidates = ocr._dedupe_crops(candidate_crops)
    unique_lines = ocr._dedupe_crops(line_candidates)
    
    print(f"Unique lines: {len(unique_lines)}")
    print(f"Unique candidates: {len(unique_candidates)}")
    
    print("\n--- Running EasyOCR on Unique Lines ---")
    for idx, crop in enumerate(unique_lines):
        local_text = ocr._read_with_easyocr(crop)
        print(f"  Line {idx} EasyOCR raw: {repr(local_text)}")
        print(f"  Line {idx} Normalized: {repr(ocr._normalize_pole_id(local_text))}")
        
    print("\n--- Running Azure Mistral on Unique Lines ---")
    for idx, crop in enumerate(unique_lines):
        raw_text = ocr._request_azure_text(crop)
        print(f"  Line {idx} Mistral raw: {repr(raw_text)}")
        print(f"  Line {idx} Normalized: {repr(ocr._normalize_pole_id(raw_text))}")
        
    print("\n--- Running Azure Mistral on Unique Candidates ---")
    for idx, crop in enumerate(unique_candidates):
        raw_text = ocr._request_azure_text(crop)
        print(f"  Candidate {idx} Mistral raw: {repr(raw_text)}")
        print(f"  Candidate {idx} Normalized: {repr(ocr._normalize_pole_id(raw_text))}")
        
if __name__ == "__main__":
    run_test()
