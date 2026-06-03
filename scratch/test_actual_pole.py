import cv2
import sys
import numpy as np
from pipeline import InfrastructurePipeline

def run_test():
    img_path = "uploads/0748854a-f0dd-4be5-98b4-3c6a54545831.jpg"
    print(f"[DEBUG] Loading image: {img_path}")
    image = cv2.imread(img_path)
    if image is None:
        print("[DEBUG] Error: image not found")
        return
        
    print("[DEBUG] Initializing InfrastructurePipeline...")
    pipeline = InfrastructurePipeline(
        comp_model="backup_models/best (2).pt",
        hardware_model="backup_models/channel_12class_v2.pt",
        shed_model="models/shed_model.pt",
        insulator_model="backup_models/insulator_model.pt"
    )
    
    # Run the predict function
    print("[DEBUG] Running pipeline.predict...")
    res = pipeline.predict(img_path, visualize=False)
    
    print("\n--- Pipeline Results ---")
    print(f"Final Class: {res.final_class}")
    print(f"Pole ID detected: {repr(res.pole_id)}")
    
    print("\n--- Detected Poles ---")
    for idx, p in enumerate(res.all_poles):
        print(f"Pole {idx}: type={p.pole_type}, conf={p.detection_conf:.3f}, box={p.box}")
        
    # Let's run process_pole_tag manually on each detected pole to see detail
    if res.all_poles:
        main_pole = max(res.all_poles, key=lambda p: (p.box[2]-p.box[0])*(p.box[3]-p.box[1]))
        print(f"\n--- Running OCR on Main Pole Box {main_pole.box} ---")
        
        # We can simulate what process_pole_tag does
        x1, y1, x2, y2 = [int(v) for v in main_pole.box]
        h_orig, w_orig = image.shape[:2]
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        
        # Print padding details
        pad_left = max(20, int(box_w * 1.8))
        pad_right = max(20, int(box_w * 0.45))
        pad_top = max(20, int(box_h * 0.06))
        pad_bottom = max(20, int(box_h * 0.08))
        print(f"Paddings: left={pad_left}, right={pad_right}, top={pad_top}, bottom={pad_bottom}")
        
        px1 = max(0, x1 - pad_left)
        py1 = max(0, y1 - pad_top)
        px2 = min(w_orig, x2 + pad_right)
        py2 = min(h_orig, y2 + pad_bottom)
        print(f"Pole crop box: [{px1}, {py1}, {px2}, {py2}] (size {px2-px1}x{py2-py1})")
        
        pole_crop = image[py1:py2, px1:px2]
        cv2.imwrite("scratch/diagnose_pole_crop.jpg", pole_crop)
        print("Saved pole crop to scratch/diagnose_pole_crop.jpg")
        
        patch_box = pipeline.ocr._find_black_patch(pole_crop)
        print(f"Patch box in pole crop: {patch_box}")
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
            print(f"Extended crop box: [{ex1}, {ey1}, {ex2}, {ey2}] (size {ex2-ex1}x{ey2-ey1})")
            cv2.imwrite("scratch/diagnose_extended_crop.jpg", extended_crop)
            print("Saved extended crop to scratch/diagnose_extended_crop.jpg")
            
            # Run OCR on extended crop
            print("\n--- Running Azure Mistral on Extended Crop ---")
            raw_text = pipeline.ocr._request_azure_text(extended_crop)
            print(f"Raw text: {repr(raw_text)}")
            print(f"Normalized: {repr(pipeline.ocr._normalize_pole_id(raw_text))}")
            
            # Let's also check other candidates
            band_crop = pipeline.ocr._find_white_text_band(extended_crop)
            if band_crop is not None:
                print(f"Band crop size: {band_crop.shape[1]}x{band_crop.shape[0]}")
                cv2.imwrite("scratch/diagnose_band_crop.jpg", band_crop)
                print("Saved band crop to scratch/diagnose_band_crop.jpg")
                raw_text_band = pipeline.ocr._request_azure_text(band_crop)
                print(f"Raw text band: {repr(raw_text_band)}")
                print(f"Normalized band: {repr(pipeline.ocr._normalize_pole_id(raw_text_band))}")

if __name__ == "__main__":
    run_test()
