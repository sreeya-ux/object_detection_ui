"""
insulator_classifier.py
────────────────────────
Classifies a detected insulator as PIN or DISC, counts sheds,
and checks for ADJUSTMENT FAULTS (misalignment/tilt).

What "adjustment fault" means here:
  - Insulator is tilted beyond acceptable angle
  - Insulator is skewed / not properly seated on crossarm
  - Detected purely from bounding box geometry + OBB angle
  - NOT physical damage (cracked/broken) — that is out of scope

Steps:
  1. Aspect ratio heuristic   → pin / disc / uncertain (fast, always runs)
  2. Crop classifier          → resolves uncertain cases
  3. Shed counter             → 11kV / 33kV / 6.3kV from pin insulator
  4. Adjustment fault check   → tilt angle vs tolerance thresholds
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from ultralytics import YOLO

from config import (
    INSULATOR_PIN_RATIO_MIN,
    INSULATOR_DISC_RATIO_MAX,
    SHED_VOLTAGE_MAP,
    SHED_VOLTAGE_GT3,
    SHED_MODEL_CONF,
    PIN_INSULATOR_IDEAL_ANGLE,
    PIN_INSULATOR_TOLERANCE_DEG,
    PIN_INSULATOR_FAULT_DEG,
    PIN_TILT_AR_THRESHOLD,
)


# ── Data class ────────────────────────────────────────────────

@dataclass
class InsulatorResult:
    box:              tuple         # (x1, y1, x2, y2)
    aspect_ratio:     float         # height / width
    type_heuristic:   str           # "pin" | "disc" | "uncertain"
    type_final:       str           # "pin" | "disc"
    type_confidence:  str           # "high" | "medium" | "low"
    shed_count:       int   = 0
    voltage:          str   = "unknown"
    detection_conf:   float = 0.0

    # Adjustment fault fields
    obb_angle_deg:    Optional[float] = None   # from OBB model
    obb_polygon:      Optional[list]  = None   # 4-point rotated polygon
    tilt_deg:         float = 0.0              # deviation from ideal vertical
    adjustment_fault: bool  = False            # True if misaligned
    fault_severity:   str   = "none"           # "none" | "warning" | "fault"
    fault_note:       str   = ""               # human-readable reason


# ── Step 1: Aspect ratio heuristic ───────────────────────────

def classify_by_aspect_ratio(
    bbox_width: float,
    bbox_height: float,
) -> tuple:
    """
    Refined heuristic based on user constraint:
    - PIN: "Stands straight" (Vertical, height > width)
    - DISC: "Sleeps on wire" (Horizontal, width > height)
    """
    if bbox_width == 0:
        return "uncertain", "low"

    ratio = bbox_height / bbox_width

    # PIN: Stands straight (Vertical)
    if ratio >= 1.1:
        confidence = "high" if ratio > 2.0 else "medium"
        return "pin", confidence

    # DISC: Sleeps on wire (Horizontal)
    elif ratio <= 0.9:
        confidence = "high" if ratio < 0.5 else "medium"
        return "disc", confidence

    else:
        # Ambiguous: nearly square
        return "uncertain", "low"


# ── Step 2: Crop classifier for uncertain cases ───────────────

class InsulatorCropClassifier:
    """
    Resolves uncertain aspect-ratio cases via:
      Option A — YOLOv11 classification model on crop (if available)
      Option B — Sobel edge texture heuristic (no model needed)

    Pin insulators have horizontal petticoat rings → more horizontal edges.
    Disc insulators are roughly circular → balanced edge distribution.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        if model_path:
            try:
                self.model = YOLO(model_path)
                print(f"Crop classifier loaded: {model_path}")
            except Exception as e:
                print(f"Crop classifier load failed: {e} — using edge heuristic")

    def classify(self, img: np.ndarray, box: tuple, padding: int = 15) -> tuple:
        crop = self._crop(img, box, padding)
        if crop is None:
            return "uncertain", "low"
        if self.model:
            return self._model_classify(crop)
        return self._edge_heuristic(crop)

    def _crop(self, img, box, pad):
        h, w = img.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        if x2 <= x1 or y2 <= y1:
            return None
        return img[y1:y2, x1:x2]

    def _model_classify(self, crop):
        results = self.model(crop, verbose=False)
        for r in results:
            if r.probs is not None:
                cls  = self.model.names[r.probs.top1].lower()
                conf = float(r.probs.top1conf)
                label = "pin" if "pin" in cls else "disc"
                return label, "high" if conf > 0.75 else "medium"
        return "uncertain", "low"

    def _edge_heuristic(self, crop):
        """
        Sobel edge energy ratio.
        Pin insulators have strong horizontal banding (petticoat rings)
        → higher horizontal edge energy.
        """
        gray   = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray   = cv2.resize(gray, (64, 128))
        sob_x  = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sob_y  = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        h_e    = np.mean(np.abs(sob_y))
        v_e    = np.mean(np.abs(sob_x))
        total  = h_e + v_e
        if total == 0:
            return "uncertain", "low"
        h_ratio = h_e / total
        if h_ratio > 0.58:
            return "pin", "medium"
        elif h_ratio < 0.42:
            return "disc", "medium"
        return "uncertain", "low"


# ── Step 3: Shed counter ──────────────────────────────────────

class ShedCounter:
    """Runs your existing shed-count model on a pin insulator crop."""

    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        print(f"Shed counter loaded: {model_path}")

    def count(self, img: np.ndarray, box: tuple, padding: int = 25) -> int:
        from config import INSULATOR_CROP_PADDING
        pad = INSULATOR_CROP_PADDING
        h, w = img.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        if x2 <= x1 or y2 <= y1:
            return 0
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return 0
        
        # ── Pre-processing: Contrast Enhancement (CLAHE) ─────
        # Apply CLAHE to Lightness channel only to preserve color features!
        # If we convert it to raw Grayscale, YOLO models trained on RGB will fail to detect.
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        crop_enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        from config import SHED_MODEL_CONF
        # iou=0.6 is crucial here: Sheds sit tightly stacked! 
        # Standard NMS (0.45) will delete overlapping sheds thinking they are duplicates.
        results = self.model(crop_enhanced, conf=SHED_MODEL_CONF, iou=0.6, verbose=False)
        count = 0
        for r in results:
            if r.boxes is not None:
                for b in r.boxes:
                    name = self.model.names[int(b.cls)].lower()
                    if any(k in name for k in ["shed","petticoat","ring","disc","groove"]):
                        count += 1
        return count

    @staticmethod
    def to_voltage(shed_count: int, insulator_type: str = "pin") -> str:
        """
        Maps disc/shed count to voltage based on user constraints.
        Rules (Indian Distribution Standard):
          11kV: 3 sheds (Pin) or 1 disc (Disc)
          33kV: >3 sheds (Pin) or 3 discs (Disc)
        """
        if insulator_type == "disc":
            if shed_count >= 3: return "33kV"
            if shed_count >= 1: return "11kV"
            return "unknown"
        
        # Pin logic
        if shed_count >= 4:
            return "33kV"
        if shed_count >= 3:
            return "11kV"
        if shed_count > 0:
            return "6.3kV" # or LT
            
        return "unknown"


# ── Step 4: Adjustment fault checker ─────────────────────────

def check_adjustment_fault(
    insulator_type: str,
    obb_angle_deg: Optional[float],
    aspect_ratio: float,
) -> tuple:
    """
    Checks whether an insulator has an adjustment (alignment) fault.

    For PIN insulators:
      - Should be near-vertical (OBB angle ≈ 90°)
      - Tilt beyond threshold → adjustment fault
      - Also checks aspect ratio: a tilted pin has lower h/w ratio

    For DISC insulators:
      - Should be near-horizontal (OBB angle ≈ 0° or 180°)
      - NOT checked here — disc string orientation is handled
        by crossarm_classifier.py (checks crossarm level instead)

    Args:
        insulator_type : "pin" or "disc"
        obb_angle_deg  : angle from OBB detection (0–90°). None = not available.
        aspect_ratio   : height / width of bbox

    Returns:
        (adjustment_fault: bool, fault_severity: str, tilt_deg: float, note: str)
        fault_severity: "none" | "warning" | "fault"
    """
    tilt_deg = 0.0

    if insulator_type != "pin":
        # Disc insulator alignment checked via crossarm level
        return False, "none", 0.0, ""

    # ── OBB angle check ───────────────────────────────────────
    if obb_angle_deg is not None:
        # For a vertical pin insulator, OBB angle ≈ 90°
        tilt_deg = abs(PIN_INSULATOR_IDEAL_ANGLE - abs(obb_angle_deg))

        if tilt_deg <= PIN_INSULATOR_TOLERANCE_DEG:
            return False, "none", tilt_deg, f"tilt={tilt_deg:.1f}° within tolerance"

        elif tilt_deg <= PIN_INSULATOR_FAULT_DEG:
            note = (
                f"Pin insulator tilted {tilt_deg:.1f}° from vertical "
                f"(tolerance ±{PIN_INSULATOR_TOLERANCE_DEG}°) — monitor"
            )
            return True, "warning", tilt_deg, note

        else:
            note = (
                f"Pin insulator tilted {tilt_deg:.1f}° from vertical — "
                f"adjustment required"
            )
            return True, "fault", tilt_deg, note

    # ── Fallback: aspect ratio check (no OBB angle available) ─
    # A properly seated pin insulator is tall → ratio > 1.5
    # A tilted one appears shorter → ratio drops
    if insulator_type == "pin" and aspect_ratio < PIN_TILT_AR_THRESHOLD:
        note = (
            f"Pin insulator aspect ratio {aspect_ratio:.2f} < "
            f"{PIN_TILT_AR_THRESHOLD} — may be tilted (no OBB angle to confirm)"
        )
        return True, "warning", tilt_deg, note

    return False, "none", tilt_deg, "ok"


# ── Combined classifier ───────────────────────────────────────

class InsulatorClassifier:
    """
    Full insulator classification + adjustment fault detection.

    Steps:
      1. Aspect ratio heuristic
      2. Crop classifier (uncertain cases only)
      3. Shed count (pin insulators only)
      4. Adjustment fault check (geometry-based, no extra model)
    """

    def __init__(
        self,
        shed_model_path: str,
        crop_classifier_path: Optional[str] = None,
    ):
        self.shed_counter    = ShedCounter(shed_model_path)
        self.crop_classifier = InsulatorCropClassifier(crop_classifier_path)

    def classify(
        self,
        img: np.ndarray,
        box: tuple,
        detection_conf: float = 0.0,
        obb_angle_deg: Optional[float] = None,
    ) -> InsulatorResult:
        """
        Classifies one insulator detection.

        Args:
            img           : full image (BGR)
            box           : (x1, y1, x2, y2)
            detection_conf: confidence from component YOLO
            obb_angle_deg : angle from OBB model (None if standard detection)
        """
        x1, y1, x2, y2 = box
        bw = x2 - x1
        bh = y2 - y1
        ar = round(bh / bw if bw > 0 else 0, 2)

        # ── Step 1: Aspect ratio heuristic ───────────────────
        type_heuristic, confidence = classify_by_aspect_ratio(bw, bh)

        # ── Step 2: Crop classifier if uncertain ─────────────
        if type_heuristic == "uncertain":
            type_final, confidence = self.crop_classifier.classify(img, box)
            if type_final == "uncertain":
                # Default to pin — more common in Indian LT/HT distribution
                type_final = "pin"
                confidence = "low"
        else:
            type_final = type_heuristic

        # ── Step 3: Shed count (Crop and run second model) ─────
        # Only run if detection confidence is high enough to warrant processing
        from config import INSULATOR_MIN_CONF
        shed_count = 0
        voltage    = "unknown"

        if detection_conf >= INSULATOR_MIN_CONF:
            # Crop and count for all types (Pin and Disc) using best_disc.pt
            shed_count = self.shed_counter.count(img, box)
            
            # SANITY OVERRIDE: 3 sheds = 11kV Pin Insulator in almost all cases.
            # If the model counted 3, we force the type to pin.
            if shed_count == 3:
                type_final = "pin"
                confidence = "high"
            
            voltage = ShedCounter.to_voltage(shed_count, type_final)
        else:
            # Fallback for low confidence (heuristic only)
            if type_final == "disc":
                voltage = "33kV"
            elif type_final == "pin":
                voltage = "11kV"

        # ── Step 4: Adjustment fault check ───────────────────
        adj_fault, severity, tilt, fault_note = check_adjustment_fault(
            type_final, obb_angle_deg, ar
        )

        return InsulatorResult(
            box             = box,
            aspect_ratio    = ar,
            type_heuristic  = type_heuristic,
            type_final      = type_final,
            type_confidence = confidence,
            shed_count      = shed_count,
            voltage         = voltage,
            detection_conf  = round(detection_conf, 2),
            obb_angle_deg   = obb_angle_deg,
            tilt_deg        = tilt,
            adjustment_fault= adj_fault,
            fault_severity  = severity,
            fault_note      = fault_note,
        )
