import os
os.chdir('/home/ubuntu/object_detection_ui')

import cv2
import numpy as np
import pytesseract
from PIL import Image
from ultralytics import YOLO
from ocr_utils import PoleOCR

def run_diagnostics(img_path):
    print(f"\n======================================")
    print(f"DIAGNOSTIC FOR: {img_path}")
    image = cv2.imread(img_path)
    if image is None:
        print("Error: Could not read image")
        return
    h_orig, w_orig = image.shape[:2]
    print(f"Original size: {w_orig}x{h_orig}")

    # Load pole detector model
    model = YOLO("models/pole_detector.pt")
    res = model(image, conf=0.1, verbose=False)
    
    ocr = PoleOCR()
    
    pole_boxes = []
    if res and res[0].boxes:
        for b_obj in res[0].boxes:
            b = b_obj.xyxy[0].cpu().numpy().astype(int)
            conf = float(b_obj.conf)
            cls = int(b_obj.cls)
            print(f"Detected pole box: {b}, conf: {conf:.2f}, class: {cls}")
            pole_boxes.append(b)
            
    if not pole_boxes:
        print("No pole boxes detected, using full image as fallback box")
        pole_boxes = [[0, 0, w_orig, h_orig]]
        
    for idx, box in enumerate(pole_boxes):
        x1, y1, x2, y2 = box
        px1 = max(0, x1 - 15)
        py1 = max(0, y1 - 15)
        px2 = min(w_orig, x2 + 15)
        py2 = min(h_orig, y2 + 15)
        pole_crop = image[py1:py2, px1:px2]
        
        # Save pole crop for inspection
        cv2.imwrite(f"diagnose_pole_{idx}.jpg", pole_crop)
        
        # 1. Test black patch detection
        patch_box = ocr._find_black_patch(pole_crop)
        print(f"Pole {idx} - Black patch detected box: {patch_box}")
        
        if patch_box:
            bx1, by1, bx2, by2 = patch_box
            # Pad slightly to keep border text intact
            bh, bw = pole_crop.shape[:2]
            pbx1 = max(0, bx1 - 8)
            pby1 = max(0, by1 - 8)
            pbx2 = min(bw, bx2 + 8)
            pby2 = min(bh, by2 + 8)
            target_crop = pole_crop[pby1:pby2, pbx1:pbx2]
        else:
            target_crop = pole_crop
            
        cv2.imwrite(f"diagnose_target_{idx}.jpg", target_crop)
        
        # Test OCR on target_crop with various preprocessing steps
        print(f"--- Running Tesseract OCR on Target Crop (Pole {idx}) ---")
        
        # Preprocessing A: Grayscale -> Resize (min_side=400) -> String
        pil_raw = ocr._to_pil(target_crop, min_side=400)
        
        # Preprocessing B: Grayscale -> Adaptive Threshold -> Resize
        gray = cv2.cvtColor(target_crop, cv2.COLOR_BGR2GRAY)
        
        # Preprocessing C: CLAHE on gray -> Otsu Threshold
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl_gray = clahe.apply(gray)
        _, otsu = cv2.threshold(cl_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        otsu_pil = ocr._to_pil(cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR), min_side=400)
        
        # Preprocessing D: Normal threshold inv
        mean_val = float(np.mean(gray))
        thresh_val = min(120, max(40, int(mean_val * 0.6)))
        _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
        thresh_pil = ocr._to_pil(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), min_side=400)
        
        # Test PSM 3 and 6 on these variations
        for label, pil_img in [("Raw (resized)", pil_raw), ("Otsu (resized)", otsu_pil), ("Thresh (resized)", thresh_pil)]:
            print(f"  [{label}]")
            for psm in [3, 6, 7, 11, 12]:
                txt = pytesseract.image_to_string(pil_img, config=f"--oem 3 --psm {psm}")
                print(f"    PSM {psm}: {repr(txt.strip())}")

run_diagnostics("/home/ubuntu/object_detection_ui/uploads/1f086938-70c5-4da4-b2dd-0540b97de2da.jpg")
run_diagnostics("/home/ubuntu/object_detection_ui/uploads/ad5df48c-27e5-45de-b304-c0a7be7cb470.jpg")
