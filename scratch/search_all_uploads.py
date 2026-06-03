import glob
import cv2
import easyocr
import os

def run_test():
    files = glob.glob("uploads/*.jpg")
    files.sort(key=os.path.getmtime, reverse=True)
    
    print("Initializing EasyOCR...")
    reader = easyocr.Reader(["en"], gpu=False)
    
    print(f"Searching {len(files)} files for RDSS / digits text...")
    for f in files[:20]:
        img = cv2.imread(f)
        if img is None:
            continue
            
        res = reader.readtext(img, detail=0)
        res_str = " | ".join(res)
        
        # Check if it has RDSS or RDS or 47 or 45
        has_match = any(token in res_str.upper() for token in ("RDSS", "RDS", "ROSS", "47", "45"))
        if has_match:
            print(f"\nMATCH FOUND: {f}")
            print(f"  Modified: {os.path.getmtime(f)}")
            print(f"  EasyOCR output: {res_str}")
        else:
            # Print if it has any text at all
            if res:
                print(f"File {f}: text = {res_str}")

if __name__ == "__main__":
    run_test()
