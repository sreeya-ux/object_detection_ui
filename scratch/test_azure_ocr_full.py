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
        
    h, w = img.shape[:2]
    print(f"Image dimensions: {w}x{h}")
    
    # Resize slightly if too large to save bandwidth, but keep it high resolution
    if max(h, w) > 1024:
        scale = 1024 / max(h, w)
        img_resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        img_resized = img
        
    print(f"Resized for OCR: {img_resized.shape[1]}x{img_resized.shape[0]}")
    
    crop_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(crop_rgb)
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=90)
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
    
    print("Sending full image to Azure Mistral OCR...")
    r = requests.post(AZURE_MISTRAL_ENDPOINT, headers=headers, json=payload, timeout=20)
    if r.status_code == 200:
        res_json = r.json()
        print("\n--- Raw JSON response (truncated keys) ---")
        # Print a nice summary of the JSON
        pages = res_json.get("pages", [])
        if pages:
            markdown = pages[0].get("markdown", "")
            print(f"Markdown Content:\n{markdown}")
            
            # Print if there is bounding box information in the JSON
            # Let's inspect some keys in pages[0]
            print("\nKeys in page dictionary:", list(pages[0].keys()))
            if "words" in pages[0]:
                print(f"Number of words detected: {len(pages[0]['words'])}")
                print("First 10 words:")
                for word in pages[0]['words'][:10]:
                    print(word)
            if "blocks" in pages[0]:
                print(f"Number of blocks: {len(pages[0]['blocks'])}")
                for block in pages[0]['blocks'][:5]:
                    print(block)
        else:
            print("No pages in response")
    else:
        print(f"Failed with status: {r.status_code}: {r.text}")

if __name__ == "__main__":
    run_test()
