import cv2
import base64
import requests
import json
from PIL import Image
from io import BytesIO
from config import AZURE_MISTRAL_API_KEY, AZURE_MISTRAL_ENDPOINT, AZURE_MISTRAL_MODEL

def run_test():
    img_path = "uploads/0748854a-f0dd-4be5-98b4-3c6a54545831.jpg"
    img = cv2.imread(img_path)
    if img is None:
        print("Image not found")
        return
        
    # Pole 1 box: (4, 794, 214, 1439)
    x1, y1, x2, y2 = 4, 794, 214, 1439
    
    # Crop with generous horizontal and vertical padding
    # Let's crop x from 0 to 450 (which covers the left side of the image)
    # and y from 750 to 1440
    crop = img[750:1440, 0:450]
    h, w = crop.shape[:2]
    print(f"Crop shape: {w}x{h}")
    cv2.imwrite("scratch/pole1_full_crop.jpg", crop)
    
    # Slice crop vertically into 4 segments
    # 1. Top: 0 to 0.35
    # 2. Upper-middle: 0.25 to 0.60
    # 3. Lower-middle: 0.50 to 0.85
    # 4. Bottom: 0.75 to 1.0
    segments = [
        ("segment_0_top", crop[0:int(h*0.35), :]),
        ("segment_1_umid", crop[int(h*0.25):int(h*0.60), :]),
        ("segment_2_lmid", crop[int(h*0.50):int(h*0.85), :]),
        ("segment_3_bot", crop[int(h*0.75):h, :])
    ]
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AZURE_MISTRAL_API_KEY}"
    }
    
    for name, seg in segments:
        seg_h, seg_w = seg.shape[:2]
        print(f"\n--- Running Azure Mistral on {name} (size: {seg_w}x{seg_h}) ---")
        cv2.imwrite(f"scratch/{name}.jpg", seg)
        
        crop_rgb = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(crop_rgb)
        buf = BytesIO()
        pil.save(buf, format="JPEG", quality=92)
        b64_data = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        payload = {
            "model": AZURE_MISTRAL_MODEL,
            "document": {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{b64_data}"
            },
            "include_image_base64": False
        }
        
        try:
            r = requests.post(AZURE_MISTRAL_ENDPOINT, headers=headers, json=payload, timeout=15)
            if r.status_code == 200:
                res_json = r.json()
                pages = res_json.get("pages", [])
                if pages:
                    markdown = pages[0].get("markdown", "")
                    print(f"Raw OCR text: {repr(markdown)}")
                else:
                    print("Empty pages list")
            else:
                print(f"Error {r.status_code}: {r.text}")
        except Exception as e:
            print(f"Failed: {e}")

if __name__ == "__main__":
    run_test()
