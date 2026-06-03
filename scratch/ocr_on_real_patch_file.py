import cv2
import base64
import requests
import json
from PIL import Image
from io import BytesIO
from ocr_utils import PoleOCR
from config import AZURE_MISTRAL_API_KEY, AZURE_MISTRAL_ENDPOINT, AZURE_MISTRAL_MODEL

def run_test():
    img_path = "uploads/6815587f-7631-4587-b510-13e5be4df393_patch.jpg"
    img = cv2.imread(img_path)
    if img is None:
        print("Image not found")
        return
        
    print(f"Patch shape: {img.shape}")
    
    ocr = PoleOCR()
    
    # 1. EasyOCR
    try:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=False)
        res_easy = reader.readtext(img, detail=0)
        print(f"EasyOCR response: {res_easy}")
        if res_easy:
            joined = " ".join(res_easy)
            print(f"EasyOCR Normalized: {repr(ocr._normalize_pole_id(joined))}")
    except Exception as e:
        print(f"EasyOCR failed: {e}")
        
    # 2. Azure Mistral
    crop_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
    
    print("\nSending patch to Azure Mistral...")
    r = requests.post(AZURE_MISTRAL_ENDPOINT, headers=headers, json=payload, timeout=10)
    if r.status_code == 200:
        res_json = r.json()
        pages = res_json.get("pages", [])
        if pages:
            markdown = pages[0].get("markdown", "")
            print(f"Mistral OCR response: {repr(markdown)}")
            print(f"Mistral Normalized: {repr(ocr._normalize_pole_id(markdown))}")
        else:
            print("Empty pages")
    else:
        print(f"Failed with status: {r.status_code}: {r.text}")

if __name__ == "__main__":
    run_test()
