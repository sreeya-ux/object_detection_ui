"""
crossarm_classifier.py
───────────────────────
Classifies crossarm shape and pole orientation.
Also checks adjustment faults — geometric misalignment only.

Adjustment faults detected here:
  - Crossarm not level (tilted beyond tolerance)
  - Pole leaning beyond acceptable angle
  - Intentional strut poles are NOT flagged as faults

No image-based fault model needed.
All checks are geometry-only (OBB angle + aspect ratio).
"""

import math
from dataclasses import dataclass
from typing import Optional
from collections import Counter

from config import (
    CROSSARM_MIN_AR_STRAIGHT,
    CROSSARM_CONDUCTOR_V_SPREAD,
    CROSSARM_T_VERTICAL_RATIO,
    CROSSARM_IDEAL_ANGLE_DEG,
    CROSSARM_TOLERANCE_DEG,
    CROSSARM_FAULT_DEG,
    POLE_IDEAL_ANGLE_DEG,
    POLE_TOLERANCE_DEG,
    POLE_FAULT_DEG,
    POLE_STRUT_THRESHOLD_DEG,
)


# ── Data classes ─────────────────────────────────────────────

@dataclass
class CrossarmResult:
    box:              tuple
    shape:            str    # "straight" | "v_arm" | "t_raising"
    confidence:       str    # "high" | "medium" | "low"
    aspect_ratio:     float
    obb_angle_deg:    Optional[float] = None
    obb_polygon:      Optional[list]  = None
    detection_conf:   float = 0.0
    # Adjustment fault
    adjustment_fault: bool  = False
    fault_severity:   str   = "none"   # "none" | "warning" | "fault"
    tilt_deg:         float = 0.0
    fault_note:       str   = ""
    note:             str   = ""


@dataclass
class PoleOrientationResult:
    box:              tuple
    pole_type:        str    # "vertical_pole" | "strut_pole"
    lean_angle_deg:   float  # degrees from vertical
    confidence:       str
    obb_polygon:      Optional[list]  = None
    detection_conf:   float = 0.0
    # Adjustment fault
    adjustment_fault: bool  = False
    fault_severity:   str   = "none"
    fault_note:       str   = ""
    note:             str   = ""


# ── Adjustment fault checkers ─────────────────────────────────

def check_crossarm_fault(obb_angle_deg: Optional[float]) -> tuple:
    """
    Checks if a crossarm is misaligned (not level).

    A correctly mounted crossarm is horizontal → OBB angle ≈ 0°.
    Tilt beyond threshold → adjustment fault.

    Returns: (fault: bool, severity: str, tilt_deg: float, note: str)
    """
    if obb_angle_deg is None:
        return False, "none", 0.0, "no OBB angle available"

    # For a horizontal crossarm, OBB angle should be near 0° (or 90° for
    # the orthogonal interpretation — depends on labeling convention).
    # We measure deviation from horizontal:
    tilt = min(abs(obb_angle_deg), abs(90 - obb_angle_deg))
    # Take the smaller of the two interpretations
    tilt = min(tilt, abs(obb_angle_deg - CROSSARM_IDEAL_ANGLE_DEG))

    if tilt <= CROSSARM_TOLERANCE_DEG:
        return False, "none", tilt, f"tilt={tilt:.1f}° within ±{CROSSARM_TOLERANCE_DEG}°"
    elif tilt <= CROSSARM_FAULT_DEG:
        return (True, "warning", tilt,
                f"Crossarm tilted {tilt:.1f}° — check mounting bolts")
    else:
        return (True, "fault", tilt,
                f"Crossarm tilted {tilt:.1f}° — adjustment required")


def check_pole_fault(
    lean_angle_deg: float,
    pole_type: str,
) -> tuple:
    """
    Checks if a pole has an adjustment fault (leaning too much).

    Strut poles lean intentionally → skip fault check for those.
    Vertical poles should be near-vertical → flag if leaning.

    Returns: (fault: bool, severity: str, note: str)
    """
    if pole_type == "strut_pole":
        # Intentional lean — not a fault
        return False, "none", f"Strut pole — intentional lean ({lean_angle_deg:.1f}°)"

    if lean_angle_deg <= POLE_TOLERANCE_DEG:
        return False, "none", f"lean={lean_angle_deg:.1f}° within tolerance"
    elif lean_angle_deg <= POLE_FAULT_DEG:
        return (True, "warning",
                f"Pole leaning {lean_angle_deg:.1f}° — monitor for progression")
    else:
        return (True, "fault",
                f"Pole leaning {lean_angle_deg:.1f}° — adjustment/re-planting required")


# ── Pole orientation classifier ───────────────────────────────

def classify_pole_orientation(
    box: tuple,
    obb_angle_deg: Optional[float] = None,
) -> PoleOrientationResult:
    """
    Classifies pole as vertical or strut, and checks for lean fault.

    Args:
        box           : (x1, y1, x2, y2)
        obb_angle_deg : from OBB detection (0–90°). None = use AR fallback.
    """
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    ar = bh / bw if bw > 0 else 0

    # ── Determine pole type + lean ────────────────────────────
    if obb_angle_deg is not None:
        lean = round(abs(POLE_IDEAL_ANGLE_DEG - abs(obb_angle_deg)), 1)

        if lean >= POLE_STRUT_THRESHOLD_DEG:
            pole_type  = "strut_pole"
            confidence = "high"
            note       = f"OBB angle={obb_angle_deg:.1f}° → intentional strut"
        elif obb_angle_deg >= 75:
            pole_type  = "vertical_pole"
            confidence = "high"
            note       = f"OBB angle={obb_angle_deg:.1f}° → vertical"
        elif obb_angle_deg >= 50:
            pole_type  = "strut_pole"
            confidence = "medium"
            note       = f"OBB angle={obb_angle_deg:.1f}° → leaning {lean}°"
        else:
            pole_type  = "strut_pole"
            confidence = "high"
            note       = f"OBB angle={obb_angle_deg:.1f}° → steep strut"
    else:
        # Fallback: Use Aspect Ratio (AR) to infer lean if OBB angle is missing
        # A leaning pole has a wider bounding box -> lower AR.
        # Vertical poles typically have AR > 5.0 (height is 5x more than width)
        if ar > 5.0:
            pole_type, confidence = "vertical_pole", "medium"
            lean = 0.0
            note = f"AR={ar:.1f} → assumed vertical"
        elif ar > 2.5:
            pole_type, confidence = "vertical_pole", "low"
            # Infer a small lean if AR starts dropping
            lean = round(max(0, (5.0 - ar) * 4), 1) 
            note = f"AR={ar:.1f} → slight lean inferred"
        else:
            pole_type, confidence = "strut_pole", "low"
            lean = 30.0
            note = f"AR={ar:.1f} → wide bbox suggests significant lean/strut"

    # ── Adjustment fault check ────────────────────────────────
    fault, severity, fault_note = check_pole_fault(lean, pole_type)

    return PoleOrientationResult(
        box              = box,
        pole_type        = pole_type,
        lean_angle_deg   = lean,
        confidence       = confidence,
        adjustment_fault = fault,
        fault_severity   = severity,
        fault_note       = fault_note,
        note             = note,
    )


# ── Crossarm shape classifier ─────────────────────────────────

def classify_crossarm_shape(
    crossarm_box: tuple,
    conductor_boxes: list,
    pole_boxes: list,
    img_shape: tuple,
    obb_angle_deg: Optional[float] = None,
    insulator_results: list = None,
    native_class: str = "",
) -> CrossarmResult:
    """
    Classifies crossarm as straight / v_arm / t_raising (or native labels).
    Also checks for crossarm alignment fault.

    Args:
        crossarm_box   : (x1, y1, x2, y2)
        conductor_boxes: list of (x1,y1,x2,y2) for conductors
        pole_boxes     : list of (x1,y1,x2,y2) for poles
        img_shape      : (height, width)
        obb_angle_deg  : from OBB detection (None = not available)
        insulator_results: list of InsulatorResult objects
        native_class   : original YOLO class string
    """
    x1, y1, x2, y2 = crossarm_box
    c_w   = x2 - x1
    c_h   = y2 - y1
    c_cx  = (x1 + x2) / 2
    ar    = c_w / c_h if c_h > 0 else 0
    img_h, img_w = img_shape[:2]

    # ── Adjustment fault check ────────────────────────────────
    fault, severity, tilt, fault_note = check_crossarm_fault(obb_angle_deg)

    # ── AI Native Label Override ──────────────────────────────
    if "t_rising" in native_class or "t_arm" in native_class:
        return CrossarmResult(
            box=crossarm_box, shape="t_raising", confidence="high",
            aspect_ratio=round(ar, 2), obb_angle_deg=obb_angle_deg,
            adjustment_fault=fault, fault_severity=severity,
            tilt_deg=tilt, fault_note=fault_note,
            note=f"Trusting YOLO Native Label: {native_class}"
        )
    if "v_cross" in native_class or "v_arm" in native_class:
        return CrossarmResult(
            box=crossarm_box, shape="v_arm", confidence="high",
            aspect_ratio=round(ar, 2), obb_angle_deg=obb_angle_deg,
            adjustment_fault=fault, fault_severity=severity,
            tilt_deg=tilt, fault_note=fault_note,
            note=f"Trusting YOLO Native Label: {native_class}"
        )
    if native_class in ["side_arm_channel", "tapping_channel"]:
        return CrossarmResult(
            box=crossarm_box, shape=native_class, confidence="high",
            aspect_ratio=round(ar, 2), obb_angle_deg=obb_angle_deg,
            adjustment_fault=fault, fault_severity=severity,
            tilt_deg=tilt, fault_note=fault_note,
            note=f"Trusting YOLO Native Label: {native_class}"
        )

    # ── T-Raising Arm (Vertical Profile) Fallback ──────────────
    # Usually vertical (AR < 2) and centered on pole.
    # High-Priority check for T-Arms as they are structurally unique.
    if pole_boxes:
        avg_pole_cx = sum((b[0]+b[2])/2 for b in pole_boxes) / len(pole_boxes)
        on_centre   = abs(c_cx - avg_pole_cx) < img_w * 0.12 # broader center
        vert_extent = c_h / img_h
        if on_centre and vert_extent > CROSSARM_T_VERTICAL_RATIO and ar < 3.5:
            return CrossarmResult(
                box=crossarm_box, shape="t_raising", confidence="medium",
                aspect_ratio=round(ar, 2), obb_angle_deg=obb_angle_deg,
                adjustment_fault=fault, fault_severity=severity,
                tilt_deg=tilt, fault_note=fault_note,
                note=f"T-arm: centred on pole, vert={vert_extent:.2%}"
            )

    # ── V-Arm (Geometric + Hardware Check) ───────────────────
    # A V-arm is physically "taller" than straight arms.
    # BROAD SEARCH: We look for insulators in a wide area around the crossarm.
    ins_ids = []
    margin_x = img_w * 0.15 # 15% image width margin
    margin_y = img_h * 0.05 # 5% image height margin
    
    if insulator_results:
        for idx, ins in enumerate(insulator_results):
            ix1, iy1, ix2, iy2 = ins.box
            icx, icy = (ix1+ix2)/2, (iy1+iy2)/2
            
            # Check containment with broad margins
            if (x1 - margin_x) <= icx <= (x2 + margin_x) and \
               (y1 - margin_y) <= icy <= (y2 + margin_y):
                ins_ids.append(idx)

    # Heuristic: V-arm has low/medium AR and contains at least 1 insulator.
    # Straight arms are usually very thin (AR > 5).
    if ar < 5.0 and len(ins_ids) >= 1:
        return CrossarmResult(
            box=crossarm_box, shape="v_arm", confidence="high",
            aspect_ratio=round(ar, 2), obb_angle_deg=obb_angle_deg,
            adjustment_fault=fault, fault_severity=severity,
            tilt_deg=tilt, fault_note=fault_note,
            note=f"V-arm: AR={ar:.2f} with {len(ins_ids)} insulators linked"
        )

    # Conductor-based fallback (for cases with high-voltage where insulators are clear)
    attached = []
    for cb in conductor_boxes:
        bx1, by1, bx2, by2 = cb
        if abs(by1 - y2) < img_h * 0.05:
            attached.append((bx1 + bx2) / 2)
    if len(attached) >= 2:
        spread = max(attached) - min(attached)
        if spread > c_w * CROSSARM_CONDUCTOR_V_SPREAD:
            return CrossarmResult(
                box=crossarm_box, shape="v_arm", confidence="medium",
                aspect_ratio=round(ar, 2), obb_angle_deg=obb_angle_deg,
                adjustment_fault=fault, fault_severity=severity,
                tilt_deg=tilt, fault_note=fault_note,
                note=f"V-arm: conductor spread={spread:.0f}px"
            )

    # ── Straight arm (default) ────────────────────────────────
    conf = "high" if ar > CROSSARM_MIN_AR_STRAIGHT else "low"
    return CrossarmResult(
        box=crossarm_box, shape="straight", confidence=conf,
        aspect_ratio=round(ar, 2), obb_angle_deg=obb_angle_deg,
        adjustment_fault=fault, fault_severity=severity,
        tilt_deg=tilt, fault_note=fault_note,
        note=f"straight: AR={ar:.1f}"
    )


# ── Aggregate multiple crossarms ──────────────────────────────

def aggregate_crossarm_results(results: list) -> tuple:
    if not results:
        return "none", 0, []
    shapes   = [r.shape for r in results]
    dominant = Counter(shapes).most_common(1)[0][0]
    faults   = [r for r in results if r.adjustment_fault]
    return dominant, len(results), faults
