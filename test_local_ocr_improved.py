#!/usr/bin/env python3
"""
Test improved local OCR for hand-painted pole IDs (RDSS 45, etc.)
Tests multiple preprocessing pipelines to find which works best.
"""
import sys
import os
import cv2
import numpy as np
import pytesseract
from PIL import Image

def test_ocr_all_methods(img_path):
    """Test multiple OCR preprocessing strategies on a pole image."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"ERROR: Could not load {img_path}")
        return
    
    print(f"Image shape: {img.shape}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    results = []

    def try_ocr(name, image_arr, config):
        try:
            text = pytesseract.image_to_string(image_arr, config=config)
            text = " ".join(text.split()).strip()
            results.append((name, text[:80] if text else "(empty)"))
            print(f"  [{name}]: {repr(text[:80]) if text else '(empty)'}")
        except Exception as e:
            print(f"  [{name}]: ERROR - {e}")

    print("\n=== Test 1: Gray PSM variants ===")
    for psm in [6, 7, 8, 11, 13]:
        try_ocr(f"gray_psm{psm}", gray, f"--oem 3 --psm {psm}")

    print("\n=== Test 2: Thresholded ===")
    _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    try_ocr("otsu_binary", binary_otsu, "--oem 3 --psm 6")
    
    inverted = cv2.bitwise_not(binary_otsu)
    try_ocr("otsu_inverted", inverted, "--oem 3 --psm 6")

    print("\n=== Test 3: Dark region crop (black paint) ===")
    # Find the darkest region - likely the black paint
    mean_val = float(np.mean(gray))
    thresh_val = min(80, max(20, int(mean_val * 0.55)))
    _, dark_mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
    
    # Morphological close to merge text+background
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_c = None
    best_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > best_area:
            bx, by, bw, bh = cv2.boundingRect(c)
            if bw > 10 and bh > 10:
                best_area = area
                best_c = (bx, by, bw, bh)
    
    if best_c:
        bx, by, bw, bh = best_c
        print(f"  Found dark patch: x={bx} y={by} w={bw} h={bh}")
        patch = img[max(0,by-5):min(h,by+bh+5), max(0,bx-5):min(w,bx+bw+5)]
        
        if patch.size > 0:
            # Scale up
            scale = max(1, 400 // max(patch.shape[:2]))
            if scale > 1:
                patch = cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            
            patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            # Invert since white text on black background
            patch_inv = cv2.bitwise_not(patch_gray)
            _, patch_bin = cv2.threshold(patch_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            try_ocr("dark_patch_inv", patch_inv, "--oem 3 --psm 6")
            try_ocr("dark_patch_bin", patch_bin, "--oem 3 --psm 6")
            try_ocr("dark_patch_psm7", patch_inv, "--oem 3 --psm 7")
            try_ocr("dark_patch_psm8", patch_inv, "--oem 3 --psm 8")
            try_ocr("dark_patch_psm11", patch_inv, "--oem 3 --psm 11")
            
            # Save patch for visual inspection
            save_path = img_path.replace('.jpg', '_patch.jpg')
            cv2.imwrite(save_path, patch)
            print(f"  Saved patch to: {save_path}")

    print("\n=== SUMMARY ===")
    for name, text in results:
        if text and text != "(empty)" and len(text) > 1:
            print(f"  {name}: {text}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_ocr_all_methods(sys.argv[1])
    else:
        # Test on all recent uploads
        uploads_dir = "/home/ubuntu/object_detection_ui/uploads"
        if os.path.exists(uploads_dir):
            files = sorted(os.listdir(uploads_dir), key=lambda x: os.path.getmtime(os.path.join(uploads_dir, x)), reverse=True)[:3]
            for f in files:
                print(f"\n{'='*60}")
                print(f"Testing: {f}")
                test_ocr_all_methods(os.path.join(uploads_dir, f))
