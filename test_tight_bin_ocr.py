import cv2
import numpy as np
import pytesseract

def clean_and_ocr():
    img_path = "/home/ubuntu/object_detection_ui/temp_inverted_patch_30.jpg"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Image not found")
        return
        
    # The image is the inverted crop. The paint patch is white/light.
    # Let's find the bounding box of the white paint patch.
    # Since the patch is light, let's threshold it at 150.
    _, patch_mask = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    
    # Clean up the mask using opening/closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    patch_mask = cv2.morphologyEx(patch_mask, cv2.MORPH_CLOSE, kernel)
    patch_mask = cv2.morphologyEx(patch_mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(patch_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found for white patch")
        return
        
    # Get largest contour
    c = max(contours, key=cv2.contourArea)
    bx, by, bw, bh = cv2.boundingRect(c)
    print(f"White patch bounding box: x={bx}, y={by}, w={bw}, h={bh}")
    
    # Crop to the white patch
    patch_crop = img[by:by+bh, bx:bx+bw]
    
    # Now, let's binarize the patch crop to get pure black text on pure white background.
    # The text is dark, background is light. Let's use Otsu or simple thresholding.
    _, binary = cv2.threshold(patch_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Add a white border (margin) around the image to help Tesseract
    margin = 20
    clean_img = cv2.copyMakeBorder(binary, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=255)
    
    # Save the cleaned image
    cv2.imwrite("/home/ubuntu/object_detection_ui/temp_clean_ocr.jpg", clean_img)
    
    # Run Tesseract OCR
    print("\n--- Running Tesseract OCR on Cleaned Image ---")
    config_whitelist = '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/- '
    for psm in [3, 4, 6, 7, 8, 11, 12]:
        text = pytesseract.image_to_string(clean_img, config=f'--oem 3 --psm {psm}')
        text_w = pytesseract.image_to_string(clean_img, config=f'--oem 3 --psm {psm} {config_whitelist}')
        print(f"PSM {psm}:")
        print(f"  Normal: {repr(text.strip())}")
        print(f"  Whitelist: {repr(text_w.strip())}")

if __name__ == '__main__':
    clean_and_ocr()
