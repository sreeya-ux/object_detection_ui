import cv2
import pytesseract
import glob

def test_uploads():
    files = glob.glob("uploads/*.jpg")
    print(f"Found {len(files)} files in uploads")
    for f in files:
        img = cv2.imread(f)
        if img is None: continue
        
        # Test raw with default PSM (3)
        text3 = pytesseract.image_to_string(img, config='--oem 3 --psm 3')
        # Test raw with PSM 6
        text6 = pytesseract.image_to_string(img, config='--oem 3 --psm 6')
        # Test raw with PSM 11
        text11 = pytesseract.image_to_string(img, config='--oem 3 --psm 11')
        
        print(f"\n=== File: {f} ===")
        print(f"PSM 3 : {repr(text3.strip())}")
        print(f"PSM 6 : {repr(text6.strip())}")
        print(f"PSM 11: {repr(text11.strip())}")

if __name__ == '__main__':
    test_uploads()
