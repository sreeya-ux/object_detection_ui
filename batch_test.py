
import os
import cv2
from ocr_utils import PoleOCR

def batch_test_here(image_folder, num_images=5):
    ocr = PoleOCR()
    images = [f for f in os.listdir(image_folder) if f.endswith('.jpg')][:num_images]
    
    print("| Image Name | Extracted Pole ID |")
    print("|------------|-------------------|")
    
    for img_name in images:
        path = os.path.join(image_folder, img_name)
        img = cv2.imread(path)
        if img is None: continue
        
        h, w = img.shape[:2]
        # Look in the middle 20% of the image vertically
        fake_box = [int(w*0.4), int(h*0.1), int(w*0.6), int(h*0.9)]
        
        result = ocr.process_pole_tag(img, fake_box)
        print(f"| {img_name} | {result} |")

if __name__ == "__main__":
    batch_test_here(r"training_data\images\val")
