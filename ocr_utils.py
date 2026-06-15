import re
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import requests
import base64


class PoleOCR:
    """
    Reads hand-painted pole IDs using Azure Mistral OCR.
    """

    def __init__(self):
        self.active = True
        self.rapidocr_reader = None
        self.easyocr_reader = None
        print("[OCR] PoleOCR ready (Azure Mistral + RapidOCR + EasyOCR).")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_black_patch(self, crop_bgr):
        """
        Find the bounding box of the darkest rectangular region — the painted
        ID area. Returns (x1, y1, x2, y2) in crop coords, or None.
        """
        try:
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]
            if h < 10 or w < 10:
                return None

            mean_val = float(np.mean(gray))
            thresh_val = min(80, max(20, int(mean_val * 0.55)))
            _, dark_mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

            contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None

            min_area = max(150, int(h * w * 0.002))
            best, best_area = None, 0

            for c in contours:
                area = cv2.contourArea(c)
                if area < min_area:
                    continue
                bx, by, bw, bh = cv2.boundingRect(c)
                if bw < 8 or bh < 8:
                    continue
                ar = max(bw, bh) / max(min(bw, bh), 1)
                if ar > 20:
                    continue
                if area > best_area:
                    best_area = area
                    best = (bx, by, bx + bw, by + bh)

            return best
        except Exception as e:
            print(f"[OCR] _find_black_patch error: {e}")
            return None

    def _enhance_for_ocr(self, crop_bgr):
        """CLAHE + sharpen so white letters on black background pop."""
        try:
            lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
            l = clahe.apply(l)
            enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
            return cv2.filter2D(enhanced, -1, kernel)
        except Exception:
            return crop_bgr

    @staticmethod
    def _dedupe_crops(crops):
        unique = []
        seen = set()
        for crop in crops:
            if crop is None or crop.size == 0:
                continue
            key = (crop.shape[0], crop.shape[1], int(np.mean(crop)), int(np.std(crop)))
            if key in seen:
                continue
            seen.add(key)
            unique.append(crop)
        return unique

    @staticmethod
    def _extract_digits(text):
        if not text:
            return ""
        cleaned = str(text).upper()
        for token in re.findall(r"[A-Z0-9}\]\)\|]+", cleaned):
            if not any(ch.isdigit() for ch in token):
                continue
            token = token.replace("O", "0").replace("Q", "0")
            token = token.replace("S", "5").replace("B", "8").replace("Z", "2")
            token = token.replace("}", "7").replace("]", "7").replace(")", "7")
            token = re.sub(r"(?<=\d)[IL]|[IL](?=\d)", "1", token)
            match = re.search(r"\d{1,4}", token)
            if match:
                return match.group(0)
        return ""

    @staticmethod
    def _looks_like_rdss(text):
        if not text:
            return False
        compact = re.sub(r"[^A-Z0-9]", "", str(text).upper())
        letters_view = compact.replace("0", "O").replace("5", "S").replace("2", "Z").replace("G", "S").replace("8", "S").replace("6", "S")
        return bool(re.search(r"R[CDO]SS", letters_view)) or "DSS" in letters_view or "RSS" in letters_view

    @staticmethod
    def _resize_for_ocr(crop_bgr, max_side=768):
        h, w = crop_bgr.shape[:2]
        longest = max(h, w)
        if longest <= max_side:
            return crop_bgr
        scale = max_side / float(longest)
        return cv2.resize(crop_bgr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)

    def _tight_crop_to_bright(self, crop_bgr):
        try:
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            bright = gray > 145
            coords = np.argwhere(bright)
            if len(coords) < 20:
                return crop_bgr
            y1, x1 = coords.min(axis=0)
            y2, x2 = coords.max(axis=0)
            h, w = gray.shape[:2]
            pad_y = max(8, h // 20)
            pad_x = max(8, w // 20)
            yy1 = max(0, int(y1) - pad_y)
            xx1 = max(0, int(x1) - pad_x)
            yy2 = min(h, int(y2) + pad_y)
            xx2 = min(w, int(x2) + pad_x)
            cropped = crop_bgr[yy1:yy2, xx1:xx2]
            return cropped if cropped.size else crop_bgr
        except Exception as e:
            print(f"[OCR] _tight_crop_to_bright error: {e}")
            return crop_bgr

    def _get_easyocr_reader(self):
        if hasattr(self, 'easyocr_reader') and self.easyocr_reader is not None:
            return self.easyocr_reader
        try:
            import easyocr
            import torch
            gpu_avail = torch.cuda.is_available()
            print(f"[OCR] Initializing EasyOCR reader (gpu={gpu_avail})...")
            self.easyocr_reader = easyocr.Reader(["en"], gpu=gpu_avail)
        except Exception as e:
            print(f"[OCR] EasyOCR unavailable: {e}")
            self.easyocr_reader = False
        return self.easyocr_reader

    def _read_with_easyocr(self, crop_bgr, use_tight=True):
        reader = self._get_easyocr_reader()
        if not reader or crop_bgr is None or crop_bgr.size == 0:
            return ""
        try:
            working_crop = crop_bgr
            if use_tight:
                working_crop = self._tight_crop_to_bright(crop_bgr)
            h, w = working_crop.shape[:2]
            longest = max(h, w)
            if longest < 220:
                scale = max(2, int(320 / longest))
                working_crop = cv2.resize(working_crop, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
            
            texts = reader.readtext(working_crop, detail=0, paragraph=True)
            clean = " ".join(t.strip() for t in texts if t and t.strip()).strip()
            if clean:
                print(f"[OCR] EasyOCR text (tight={use_tight}): '{clean}'")
            return clean
        except Exception as e:
            print(f"[OCR] EasyOCR error: {e}")
            return ""

    def _get_rapidocr_reader(self):
        if hasattr(self, 'rapidocr_reader') and self.rapidocr_reader is not None:
            return self.rapidocr_reader
        try:
            try:
                from rapidocr import RapidOCR
            except ImportError:
                from rapidocr_onnxruntime import RapidOCR
            self.rapidocr_reader = RapidOCR()
        except Exception as e:
            print(f"[OCR] RapidOCR unavailable: {e}")
            self.rapidocr_reader = False
        return self.rapidocr_reader

    def _read_with_rapidocr(self, crop_bgr, use_tight=True):
        reader = self._get_rapidocr_reader()
        if not reader or crop_bgr is None or crop_bgr.size == 0:
            return ""
        try:
            working_crop = crop_bgr
            if use_tight:
                working_crop = self._tight_crop_to_bright(crop_bgr)
            h, w = working_crop.shape[:2]
            longest = max(h, w)
            if longest > 384:
                scale = 384.0 / float(longest)
                working_crop = cv2.resize(working_crop, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
            elif longest < 220:
                scale = max(2, int(320 / longest))
                working_crop = cv2.resize(working_crop, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
            
            result, _ = reader(working_crop)
            pieces = []
            if result:
                for line in result:
                    if not line or len(line) < 2:
                        continue
                    text = str(line[1]).strip()
                    if text:
                        pieces.append(text)
            clean = " ".join(pieces).strip()
            if clean:
                print(f"[OCR] RapidOCR text (tight={use_tight}): '{clean}'")
            return clean
        except Exception as e:
            print(f"[OCR] RapidOCR error: {e}")
            return ""

    @staticmethod
    def _pole_score(text):
        """Score a text string: higher = more likely a real pole ID."""
        normalized = PoleOCR._normalize_pole_id(text)
        if normalized == "Not Found":
            return -9999
        
        # High base score for valid normalized pole ID
        s = 1000
        
        # Give higher score for more specific matches
        if normalized.startswith("RDSS"):
            s += 500
            # Reward more specific numeric suffixes, and penalize fragile 1-digit IDs.
            digit_groups = re.findall(r"\d+", normalized)
            if digit_groups:
                longest_group = max(len(group) for group in digit_groups)
                s += sum(len(group) for group in digit_groups) * 100
                if longest_group == 1:
                    s -= 250
                elif longest_group >= 2:
                    s += 300
            
            # Special reward for HT / LT suffixes
            if "HT" in normalized or "LT" in normalized:
                s += 200
        else:
            # Generic pattern (e.g. AB 123)
            s += 200
            digits = re.findall(r"\d", normalized)
            s += len(digits) * 50
            
        return s

    @staticmethod
    def _is_valid_id(val):
        if not val or val == "Not Found":
            return False
        # Allow if it has digits, or if it is "RDSS", "RDSS HT", "RDSS LT"
        return any(ch.isdigit() for ch in val) or val.startswith("RDSS")

    @staticmethod
    def _normalize_pole_id(text):
        """Extract painted pole IDs like RDSS 45 from noisy OCR output."""
        if not text:
            return "Not Found"

        raw = str(text).upper()
        raw = raw.replace("|", "1").replace("!", "1")
        raw = raw.replace("₹", "").replace("¥", "").replace("`", "")
        # Replace common OCR misreadings of brackets/braces/symbols before stripping
        raw = raw.replace("}", "7").replace("]", "7").replace(")", "7")
        raw = raw.replace("{", "6").replace("[", "6").replace("(", "6")
        raw = raw.replace("+", "7")
        
        # Split by spaces and clean each word
        words = raw.split()
        cleaned_words = []
        for w in words:
            cw = re.sub(r"[^A-Z0-9]", "", w)
            if cw:
                cleaned_words.append(cw)
        
        compact = "".join(cleaned_words)
        if len(compact) < 2:
            return "Not Found"

        letters_view = compact.replace("0", "O").replace("5", "S").replace("2", "Z").replace("G", "S").replace("8", "S").replace("6", "S")
        
        # Support common OCR misreadings of RDSS: RDSS, RCSS, ROSS
        rdss_match = re.search(r"R[CDO]SS", letters_view)
        if rdss_match:
            after = compact[rdss_match.end():]
            
            # Check for HT / LT suffix (common on Indian poles)
            has_ht = "HT" in after or "H7" in after
            has_lt = "LT" in after or "L1" in after
            
            # Remove HT/LT from digit extraction
            after_digits_only = after.replace("HT", "").replace("H7", "").replace("LT", "").replace("L1", "")
            
            after_cleaned = after_digits_only.replace("O", "0").replace("Q", "0")
            after_cleaned = re.sub(r"(?<=\d)[IL]|[IL](?=\d)", "1", after_cleaned)
            after_cleaned = after_cleaned.replace("S", "5").replace("B", "8").replace("Z", "2")
            
            digits = re.search(r"\d{1,4}", after_cleaned)
            
            suffix = ""
            if has_ht:
                suffix = " HT"
            elif has_lt:
                suffix = " LT"
                
            if digits:
                return f"RDSS {digits.group(0)}{suffix}"
            elif suffix:
                return f"RDSS{suffix}"
            return "RDSS"

        # Try matching generic prefix + suffix (e.g. AB 123)
        generic = re.search(r"([A-Z]{2,6})(\d{1,4})", compact)
        if generic:
            prefix = generic.group(1)
            # Correct RCSS / ROSS / RDSS prefixes without digit suffix in generic match
            if prefix in ("RCSS", "ROSS", "RDSG", "ROSG", "RCSG"):
                prefix = "RDSS"
                
            # Reject generic OCR hallucinations on non-text regions
            if prefix in ("ID", "IV", "IMG", "IP", "IMAGE", "PAGE", "FIG", "TABLE"):
                return "Not Found"
                
            return f"{prefix} {generic.group(2)}"

        # If it has digits, fallback to returning the cleaned words directly
        # with spaces cleanly separated between letters and digits.
        fallback = " ".join(cleaned_words)
        fallback = re.sub(r"([A-Z]+)(\d+)", r"\1 \2", fallback)
        fallback = re.sub(r"(\d+)([A-Z]+)", r"\1 \2", fallback)
        fallback_digits = re.search(r"\d+", fallback.replace("O", "0").replace("Q", "0").replace("S", "5"))
        if fallback_digits:
            return fallback

        return "Not Found"

    def _candidate_crops(self, pole_crop, patch_crop):
        """Return focused crops, including lower painted-band slices."""
        crops = []
        for crop in [patch_crop, pole_crop]:
            if crop is None or crop.size == 0:
                continue
            crops.append(crop)
            h, w = crop.shape[:2]
            if h >= 30 and w >= 30:
                crops.append(crop[int(h * 0.30):h, :])
                crops.append(crop[int(h * 0.45):h, :])
                crops.append(crop[int(h * 0.55):h, :])

        unique = []
        seen = set()
        for crop in crops:
            if crop is None or crop.size == 0:
                continue
            key = crop.shape[:2]
            if key not in seen:
                unique.append(crop)
                seen.add(key)
        return unique

    def _extract_text_line_crops(self, crop_bgr):
        """Split the painted patch into likely text lines using local image analysis."""
        try:
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]
            if h < 40 or w < 40:
                return []

            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            bright_mask = (gray > 150).astype(np.uint8) * 255
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bright_mask, connectivity=8)
            boxes = []
            for idx in range(1, num_labels):
                x, y, bw, bh, area = stats[idx]
                if area < max(24, (h * w) // 1200):
                    continue
                if bh < max(8, h // 24):
                    continue
                if bw < max(8, w // 28):
                    continue
                # Ignore long, thin underline strokes that connect the two lines.
                if bw > int(w * 0.35) and bh < max(14, h // 14):
                    continue
                boxes.append((x, y, x + bw, y + bh))

            if not boxes:
                return []

            boxes.sort(key=lambda b: (b[1] + b[3]) / 2.0)
            groups = [[boxes[0]]]
            y_gap = max(18, h // 10)
            for box in boxes[1:]:
                cy = (box[1] + box[3]) / 2.0
                prev_group = groups[-1]
                prev_cy = np.mean([(b[1] + b[3]) / 2.0 for b in prev_group])
                if abs(cy - prev_cy) <= y_gap:
                    prev_group.append(box)
                else:
                    groups.append([box])

            line_crops = []
            for group in groups[:3]:
                x1 = min(b[0] for b in group)
                y1 = min(b[1] for b in group)
                x2 = max(b[2] for b in group)
                y2 = max(b[3] for b in group)
                pad_x = max(10, w // 24)
                pad_y = max(8, h // 24)
                xx1 = max(0, x1 - pad_x)
                yy1 = max(0, y1 - pad_y)
                xx2 = min(w, x2 + pad_x)
                yy2 = min(h, y2 + pad_y)
                line = self._tight_crop_to_bright(crop_bgr[yy1:yy2, xx1:xx2])
                if line.size:
                    line_crops.append(line)

            return self._dedupe_crops(line_crops[:3])
        except Exception as e:
            print(f"[OCR] _extract_text_line_crops error: {e}")
            return []

    def _fractional_line_crops(self, crop_bgr):
        """Fallback split for common two-line pole tags: prefix on top, digits below."""
        try:
            h, w = crop_bgr.shape[:2]
            if h < 60 or w < 60:
                return []
            top = self._tight_crop_to_bright(crop_bgr[max(0, int(h * 0.06)):max(1, int(h * 0.58)), :])
            bottom = self._tight_crop_to_bright(crop_bgr[max(0, int(h * 0.46)):min(h, int(h * 0.98)), :])
            return self._dedupe_crops([top, bottom])
        except Exception as e:
            print(f"[OCR] _fractional_line_crops error: {e}")
            return []

    def _find_white_text_band(self, crop_bgr):
        """Find a dark painted band with bright text, common on pole IDs."""
        try:
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]
            if h < 40 or w < 40:
                return None

            dark = gray < 95
            bright = gray > 145
            row_score = dark.mean(axis=1) + (bright.mean(axis=1) * 0.7)
            good_rows = np.where(row_score > 0.18)[0]
            if len(good_rows) < 10:
                return None

            y1, y2 = int(good_rows[0]), int(good_rows[-1])
            y1 = max(0, y1 - 20)
            y2 = min(h, y2 + 20)
            band = crop_bgr[y1:y2, :]
            return band if band.size else None
        except Exception:
            return None

    def _read_with_gemini(self, crop_bgr):
        """Use Gemini 2.5 Flash to read the hand-painted pole tag from the crop."""
        from config import GEMINI_API_KEY, USE_LLM_OCR
        if not USE_LLM_OCR or not GEMINI_API_KEY or crop_bgr is None or crop_bgr.size == 0:
            return ""

        try:
            # Resize image if too large to save bandwidth
            working_crop = self._resize_for_ocr(crop_bgr, max_side=512)
            # Encode BGR image to JPEG bytes
            success, encoded_img = cv2.imencode('.jpg', working_crop)
            if not success:
                return ""
            img_bytes = encoded_img.tobytes()

            # Initialize Gemini client
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=GEMINI_API_KEY)

            prompt = (
                "You are an expert OCR engine. Read the hand-painted pole identification number/tag from this image. "
                "The tag usually starts with 'RDSS' (or similar misreadings like ROSS, RCSS, RDSS HT, RDSS LT) followed by a number (e.g., RDSS 84, RDSS 5, etc.). "
                "Only return the raw text representing the pole ID (e.g., 'RDSS 84'). If no pole ID is visible, return 'Not Found'."
            )

            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[
                    types.Part.from_bytes(
                        data=img_bytes,
                        mime_type='image/jpeg',
                    ),
                    prompt
                ]
            )

            text = response.text.strip()
            print(f"[OCR] Gemini raw output: '{text}'")
            return text

        except Exception as e:
            print(f"[OCR] Gemini API error: {e}")
            return ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def has_black_paint(self, crop):
        """Legacy compatibility — kept for callers that use this."""
        return self._find_black_patch(crop) is not None

    def process_pole_tag(self, image, box):
        """
        Read the hand-painted ID on a detected pole using local EasyOCR.

        image : BGR numpy array (full original image)
        box   : [x1, y1, x2, y2] in image pixel coordinates
        Returns: pole ID string, or 'Not Found'
        """
        if not self.active:
            return "Not Found"

        try:
            x1, y1, x2, y2 = [int(v) for v in box]
            h_orig, w_orig = image.shape[:2]
            box_w = max(1, x2 - x1)
            box_h = max(1, y2 - y1)

            # Pole detections often hug only one side of the concrete pole.
            # Expand much farther to the left so we can recover the painted tag.
            pad_left = max(20, int(box_w * 1.8))
            pad_right = max(20, int(box_w * 0.45))
            pad_top = max(20, int(box_h * 0.06))
            pad_bottom = max(20, int(box_h * 0.08))

            px1 = max(0, x1 - pad_left)
            py1 = max(0, y1 - pad_top)
            px2 = min(w_orig, x2 + pad_right)
            py2 = min(h_orig, y2 + pad_bottom)
            pole_crop = image[py1:py2, px1:px2]

            if pole_crop.size == 0:
                return "Not Found"

            candidate_crops = []
            line_candidates = []

            # Try to zoom into the black painted patch first
            patch_box = self._find_black_patch(pole_crop)
            if patch_box:
                bx1, by1, bx2, by2 = patch_box
                ph, pw = pole_crop.shape[:2]
                pad_x = max(18, pw // 20)
                pad_top = max(18, ph // 20)
                pad_bottom = max(40, ph // 8)
                ex1 = max(0, bx1 - pad_x)
                ey1 = max(0, by1 - pad_top)
                ex2 = min(pw, bx2 + pad_x)
                ey2 = min(ph, by2 + pad_bottom)
                extended_crop = pole_crop[ey1:ey2, ex1:ex2]
                if extended_crop.size > 0:
                    candidate_crops.append(extended_crop)
                    band_crop = self._find_white_text_band(extended_crop)
                    if band_crop is not None and band_crop.size > 0:
                        candidate_crops.append(band_crop)
                        line_candidates.extend(self._extract_text_line_crops(band_crop))
                        line_candidates.extend(self._fractional_line_crops(band_crop))
                    else:
                        line_candidates.extend(self._extract_text_line_crops(extended_crop))
                        line_candidates.extend(self._fractional_line_crops(extended_crop))
                print("[OCR] Local tag crop analysis prepared")
            else:
                print("[OCR] No black patch — using full pole crop")

            if not candidate_crops:
                candidate_crops.append(pole_crop)
            elif len(candidate_crops) < 2:
                fallback_band = self._find_white_text_band(candidate_crops[0])
                candidate_crops.append(fallback_band if fallback_band is not None and fallback_band.size > 0 else pole_crop)

            unique_candidates = self._dedupe_crops(candidate_crops)
            # Also dedupe line-level crops (were built but previously never used)
            unique_line_candidates = self._dedupe_crops(line_candidates)

            best_result = "Not Found"
            best_score = -9999
            raw_texts_seen = []  # track all raw OCR texts for multi-line combination

            # 1. Try EasyOCR first on full-patch candidates (if available)
            easy_reader = self._get_easyocr_reader()
            if easy_reader:
                for crop in unique_candidates:
                    local_text = self._read_with_easyocr(crop, use_tight=False)
                    if local_text:
                        raw_texts_seen.append(local_text)
                        normalized = self._normalize_pole_id(local_text)
                        if normalized != "Not Found":
                            score = self._pole_score(normalized) + 200  # Extra weight for EasyOCR
                            if score > best_score:
                                best_score = score
                                best_result = normalized

            # 2. Try RapidOCR on full-patch candidates (as secondary/complement)
            for crop in unique_candidates:
                local_text = self._read_with_rapidocr(crop, use_tight=False)
                if local_text:
                    raw_texts_seen.append(local_text)
                    normalized = self._normalize_pole_id(local_text)
                    if normalized != "Not Found":
                        score = self._pole_score(normalized)
                        if score > best_score:
                            best_score = score
                            best_result = normalized

            # 3. Try line-level crops (top/bottom splits) — fixes two-line "RDSS\n84" tags
            for crop in unique_line_candidates:
                if easy_reader:
                    local_text = self._read_with_easyocr(crop, use_tight=True)
                    if local_text:
                        raw_texts_seen.append(local_text)
                        normalized = self._normalize_pole_id(local_text)
                        if normalized != "Not Found":
                            score = self._pole_score(normalized) + 200
                            if score > best_score:
                                best_score = score
                                best_result = normalized

                local_text = self._read_with_rapidocr(crop, use_tight=True)
                if local_text:
                    raw_texts_seen.append(local_text)
                    normalized = self._normalize_pole_id(local_text)
                    if normalized != "Not Found":
                        score = self._pole_score(normalized)
                        if score > best_score:
                            best_score = score
                            best_result = normalized

            # 4. Multi-line combination: if we have "RDSS" but no digits yet,
            # hunt for a standalone digit in ALL raw OCR texts seen so far.
            if best_result == "RDSS" or (best_result.startswith("RDSS") and not re.search(r"\d", best_result)):
                print("[OCR] RDSS prefix found without digits — scanning line crops for number...")
                for raw in raw_texts_seen:
                    m = re.search(r"\b(\d{1,4})\b", raw)
                    if m:
                        candidate = f"RDSS {m.group(1)}"
                        score = self._pole_score(candidate)
                        if score > best_score:
                            best_score = score
                            best_result = candidate
                            print(f"[OCR] Multi-line combination: '{candidate}' (score={score})")
                # Also try reading each line crop for pure digits
                for crop in unique_line_candidates:
                    if easy_reader:
                        line_text = self._read_with_easyocr(crop, use_tight=False)
                        if line_text:
                            m = re.search(r"\b(\d{1,4})\b", line_text)
                            if m:
                                candidate = f"RDSS {m.group(1)}"
                                score = self._pole_score(candidate) + 200
                                if score > best_score:
                                    best_score = score
                                    best_result = candidate
                                    print(f"[OCR] EasyOCR line crop digit hunt: '{candidate}' (score={score})")
                    
                    line_text = self._read_with_rapidocr(crop, use_tight=False)
                    if line_text:
                        m = re.search(r"\b(\d{1,4})\b", line_text)
                        if m:
                            candidate = f"RDSS {m.group(1)}"
                            score = self._pole_score(candidate)
                            if score > best_score:
                                best_score = score
                                best_result = candidate
                                print(f"[OCR] RapidOCR line crop digit hunt: '{candidate}' (score={score})")

            # 5. Fallback: try Gemini OCR on unique candidates if local OCR results are weak or Not Found
            if best_result == "Not Found" or best_score < 1500:
                print(f"[OCR] Local OCR score is low ({best_score}) or Not Found. Trying Gemini OCR fallback...")
                for crop in unique_candidates[:2]:
                    gemini_text = self._read_with_gemini(crop)
                    if gemini_text:
                        normalized = self._normalize_pole_id(gemini_text)
                        if normalized != "Not Found":
                            score = self._pole_score(normalized) + 500  # Extra weight for Gemini
                            if score > best_score:
                                best_score = score
                                best_result = normalized
                                print(f"[OCR] Gemini found high-confidence ID: '{best_result}' (score={best_score})")

            if best_result != "Not Found":
                print(f"[OCR] Selected best OCR ID: '{best_result}' (score={best_score})")
                return best_result

            return "Not Found"

        except Exception as e:
            print(f"[OCR] process_pole_tag error: {e}")
            return "Not Found"
