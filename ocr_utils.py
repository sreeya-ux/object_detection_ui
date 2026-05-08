
import cv2
import numpy as np

try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False

class PoleOCR:
    def __init__(self):
        if HAS_EASYOCR:
            print("Initializing OCR Engine (EasyOCR)...")
            # gpu=False for better compatibility on this machine
            self.reader = easyocr.Reader(['en'], gpu=False)
        else:
            print("EasyOCR not found. Running in detection-only mode.")

    def process_pole_tag(self, image, box):
        """
        Processes a detected pole: Crops, Enhances, and Scans for IDs.
        box: [x1, y1, x2, y2]
        """
        if not HAS_EASYOCR:
            return "OCR Not Installed"

        x1, y1, x2, y2 = box
        # 1. Crop the pole area (add a tiny margin)
        h_orig, w_orig = image.shape[:2]
        x1, y1 = max(0, x1-5), max(0, y1-5)
        x2, y2 = min(w_orig, x2+5), min(h_orig, y2+5)
        crop = image[y1:y2, x1:x2]
        
        if crop.size == 0:
            return "No Pole Area"

        # 2. Pre-process: Slicing into Tiles
        h, w = crop.shape[:2]
        # Instead of rotating, we'll slice the vertical pole into square chunks
        # This is better for upright letters on a vertical pole
        tile_size = w * 2 # Make tiles twice as wide as the pole
        if tile_size < 50: tile_size = 100 # Ensure tiles aren't too tiny
        overlap = int(tile_size * 0.3)
        
        found_text = []
        for y in range(0, h, tile_size - overlap):
            y_end = min(y + tile_size, h)
            tile = crop[y:y_end, 0:w]
            
            # Pre-boost contrast on color tile before grayscale conversion
            tile_lab = cv2.cvtColor(tile, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(tile_lab)
            clahe_color = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe_color.apply(l)
            tile = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
            
            # 1. Upscale 2x for better character detail
            tile = cv2.resize(tile, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            
            # --- TRIPLE-CHECK CONSENSUS SYSTEM ---
            gray_tile = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
            
            # Pass 1: Denoised
            denoised = cv2.fastNlMeansDenoising(gray_tile, None, 10, 7, 21)
            res1 = self.reader.readtext(denoised, detail=0)
            
            # Pass 2: High Contrast (CLAHE)
            clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            res2 = self.reader.readtext(enhanced, detail=0)
            
            # Pass 3: Sharpened
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            res3 = self.reader.readtext(sharpened, detail=0)
            
            # Combine all unique findings from all 3 passes
            for text in (res1 + res2 + res3):
                text = text.strip()
                if len(text) > 1:
                    found_text.append(text)
            
            if y_end == h: break

        # 3. Clean up duplicates (since tiles overlap)
        unique_text = []
        for t in found_text:
            # --- SMART CLEANER: Fix common misreads for power poles ---
            t_clean = t.upper()
            t_clean = t_clean.replace("ROSS", "RDSS")
            t_clean = t_clean.replace("RVSS", "RDSS")
            t_clean = t_clean.replace("RZSS", "RDSS")
            t_clean = t_clean.replace("S5", "SS-7")
            t_clean = t_clean.replace("LT", "Lt") # User prefers 'Lt'
            
            if not unique_text or t_clean != unique_text[-1]:
                unique_text.append(t_clean)
            
        return " ".join(unique_text) if unique_text else "Not Found"
