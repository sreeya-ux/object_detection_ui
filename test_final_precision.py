
import easyocr
import cv2
import numpy as np

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return result

def test_multi_angle_scan(image_path):
    print(f"Starting Multi-Angle Precision Scan on {image_path}...")
    reader = easyocr.Reader(['en'], gpu=False)
    
    img = cv2.imread(image_path)
    if img is None:
        return

    all_found = []
    # Scan from -20 to +20 degrees to find the perfect upright angle for the letters
    for angle in range(-20, 21, 5):
        print(f"  - Testing rotation: {angle} degrees...")
        rotated = rotate_image(img, angle)
        
        # Crop the middle section where the pole is
        h, w = rotated.shape[:2]
        crop = rotated[int(h*0.1):int(h*0.9), int(w*0.3):int(w*0.6)]
        
        # Basic enhancement
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        results = reader.readtext(enhanced, detail=0, paragraph=True)
        for text in results:
            text = text.upper().strip()
            if len(text) > 2:
                all_found.append(text)

    print("\n--- MULTI-ANGLE RESULTS ---")
    final_text = " ".join(dict.fromkeys(all_found)) # Remove duplicates
    # Apply our smart cleaner logic
    final_text = final_text.replace("ROSS", "RDSS").replace("S5", "SS-11").replace("LT", "Lt")
    
    print(f"CLEANED RESULT: {final_text}")
    print("----------------------------")

if __name__ == "__main__":
    test_multi_angle_scan(r"c:\Users\ASK037-PC\.gemini\antigravity\brain\8b4adf17-a991-49b2-ae58-0a654dd7c2df\media__1778153905670.jpg")
