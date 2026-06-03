import cv2
import easyocr
import re

def run_test():
    img_path = "uploads/0748854a-f0dd-4be5-98b4-3c6a54545831.jpg"
    img = cv2.imread(img_path)
    if img is None:
        print("Image not found")
        return
        
    reader = easyocr.Reader(["en"], gpu=False)
    
    # Pole 0: (437, 60, 691, 1434)
    # Pole 1: (4, 794, 214, 1439)
    
    poles = [
        {"name": "Pole 0", "box": (437, 60, 691, 1434)},
        {"name": "Pole 1", "box": (4, 794, 214, 1439)}
    ]
    
    for p in poles:
        x1, y1, x2, y2 = p["box"]
        crop = img[y1:y2, x1:x2]
        cv2.imwrite(f"scratch/crop_{p['name'].replace(' ', '_').lower()}.jpg", crop)
        print(f"\n--- Running EasyOCR on {p['name']} (box: {p['box']}, size: {crop.shape}) ---")
        
        # Test 1: Full crop
        res = reader.readtext(crop, detail=0)
        print(f"Full crop text: {res}")
        
        # Test 2: Standard padding
        box_w = x2 - x1
        box_h = y2 - y1
        pad_left = max(20, int(box_w * 1.8))
        pad_right = max(20, int(box_w * 0.45))
        pad_top = max(20, int(box_h * 0.06))
        pad_bottom = max(20, int(box_h * 0.08))
        
        px1 = max(0, x1 - pad_left)
        py1 = max(0, y1 - pad_top)
        px2 = min(img.shape[1], x2 + pad_right)
        py2 = min(img.shape[0], y2 + pad_bottom)
        
        pad_crop = img[py1:py2, px1:px2]
        cv2.imwrite(f"scratch/pad_crop_{p['name'].replace(' ', '_').lower()}.jpg", pad_crop)
        print(f"Padded crop box: {px1, py1, px2, py2}, size: {pad_crop.shape}")
        
        res_pad = reader.readtext(pad_crop, detail=0)
        print(f"Padded crop text: {res_pad}")
        
        # Draw bounding boxes and text on padded crop to see what's happening
        # Also run EasyOCR with details
        res_pad_detail = reader.readtext(pad_crop, detail=1)
        print("Padded crop details:")
        for bbox, text, conf in res_pad_detail:
            print(f"  - '{text}' (conf: {conf:.2f}) at {bbox}")

if __name__ == "__main__":
    run_test()
