"""
pipeline.py
────────────
Orchestrates all modules into one inference call.

Loads:
  - OBB/standard YOLO component model
  - Your existing shed-count model
  - Optional: lightweight insulator crop classifier

On each image:
  1. Run component YOLO → detect insulators, poles, crossarms, conductors
  2. For each insulator:
       a. Aspect ratio heuristic (fast)
       b. Crop classifier if uncertain
       c. Shed count model (your model) on confirmed pin insulators
  3. Classify crossarm shapes (straight / V / T)
  4. Classify pole orientation (vertical / strut)
  5. Detect HT+LT combined
  6. Rule engine → final pole class

Usage:
  pipeline = InfrastructurePipeline("component.pt", "shed.pt")
  result   = pipeline.predict("field_photo.jpg")
  print(result.final_class, result.reason)
"""

import cv2
import math
import numpy as np
import sys
from pathlib import Path

# Fix relative imports when running from root
curr_dir = Path(__file__).parent.absolute()
if str(curr_dir) not in sys.path:
    sys.path.append(str(curr_dir))

from typing import List, Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter

from ultralytics import YOLO

from config import (
    DETECTION_CONF, DETECTION_IOU, HT_LT_HEIGHT_THRESHOLD,
    OBB_CLASS_KEYWORDS, POLE_CLASSES,
    THRESHOLD_INSULATOR, THRESHOLD_CROSSARM, THRESHOLD_POLE, THRESHOLD_CONDUCTOR,
    GLOBAL_TILT_MAX_DEG
)
from insulator_classifier import InsulatorClassifier, InsulatorResult
from crossarm_classifier  import (
    classify_pole_orientation, classify_crossarm_shape,
    aggregate_crossarm_results, PoleOrientationResult, CrossarmResult
)
from rule_engine import classify_pole, ComponentSignals, ClassificationResult
# OCR removed as requested
# from ocr_utils import PoleOCR


# ── Pipeline output ───────────────────────────────────────────

@dataclass
class PipelineResult:
    """Complete output of one inference run."""
    final_class:      str
    class_id:         int
    reason:           str
    voltage:          str
    confidence:       str
    signals_used:     list

    insulators:       list  = field(default_factory=list)
    all_poles:        list  = field(default_factory=list) # List of PoleOrientationResult
    pole_orientation: Optional[PoleOrientationResult] = None # Primary pole for rule engine
    crossarms:        list  = field(default_factory=list)
    conductors:       list  = field(default_factory=list) # Actual boxes
    street_lights:    list  = field(default_factory=list) # (box, conf, poly)
    others:           list  = field(default_factory=list) # (label, box, conf, poly)
    crossarm_shape:   str   = "none"
    crossarm_count:   int   = 0
    conductor_count:  int   = 0
    pole_id:          str   = "Not Found"
    flags:            dict  = field(default_factory=dict)

    # Adjustment fault summary
    adjustment_faults: list = field(default_factory=list)
    # Each entry: {"component": str, "severity": str, "note": str}


# ── Helper: keyword matcher ───────────────────────────────────

def _match_keyword(cls_name: str, category: str) -> bool:
    """
    Checks if a detected class name belongs to a category.
    Uses OBB_CLASS_KEYWORDS from config.
    """
    name_lower = cls_name.lower()
    
    # Safety guard: 'clamp' contains the substring 'lamp', but a clamp is never a lamp/street light
    if category == "lamp_head" and "clamp" in name_lower:
        return False
        
    # Safety guard: 'street' contains 'tree', but a street light is never vegetation
    if category == "vegetation" and "street" in name_lower:
        return False
        
    return any(kw.lower() in name_lower for kw in OBB_CLASS_KEYWORDS.get(category, []))


# ── Main pipeline ─────────────────────────────────────────────

class InfrastructurePipeline:
    """
    Full 33kV infrastructure inspection pipeline.
    Combine component detection + angle classification + rule engine.
    """

    def __init__(
        self,
        component_model_path: str,
        insulator_model_path: str,
        shed_model_path: str,
        crossarm_model_path: Optional[str] = None,
        crop_classifier_path: Optional[str] = None,
        conf: float = DETECTION_CONF,
        iou:  float = DETECTION_IOU,
    ):
        """
        Args:
            component_model_path : path to your trained component YOLO (.pt)
            insulator_model_path : path to specialized insulator detection model (.pt)
            shed_model_path      : path to your shed-count model (.pt)
            crossarm_model_path  : path to model for crossarms (e.g., best_whole.pt)
            crop_classifier_path : path to optional insulator crop classifier (.pt)
            conf                 : detection confidence threshold
            iou                  : NMS IoU threshold
        """
        print("Loading component model...")
        self.component_model = YOLO(component_model_path)
        # Check model task directly (OBB models return 'obb')
        self.is_obb = self.component_model.task == 'obb'

        print("Loading dedicated insulator detector...")
        self.insulator_detector = YOLO(insulator_model_path)

        if crossarm_model_path and Path(crossarm_model_path).exists():
            print("Loading dedicated crossarm detector...")
            self.crossarm_detector = YOLO(crossarm_model_path)
        else:
            self.crossarm_detector = None

        print("Loading insulator classifier (shed model + crop classifier)...")
        self.insulator_clf = InsulatorClassifier(
            shed_model_path      = shed_model_path,
            crop_classifier_path = crop_classifier_path,
        )

        self.conf = conf
        self.iou  = iou
        
        self.ocr_engine = None

        print("Pipeline ready.\n")

    def predict(
        self,
        image_path: str,
        visualize: bool = True,
        save_path: Optional[str] = None,
    ) -> PipelineResult:
        """
        Runs full pipeline on one image.

        Args:
            image_path : path to input image
            visualize  : draw and save annotated output image
            save_path  : where to save visualization (None = auto)

        Returns:
            PipelineResult with final classification + all details
        """
        img_original = cv2.imread(image_path)
        if img_original is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        # ── Pre-processing: Image Enhancement ──────────────────
        img = self._enhance_image(img_original)

        img_h, img_w = img.shape[:2]

        import gc
        import torch

        # ── Step 1: Run component detector (Optimized Single Scale) ────
        # 800px serves as a high-stability resolution for 1.7GB RAM environments.
        # This significantly reduces memory spikes and prevents NGrok timeouts.
        raw_combined_res = self.component_model(image_path, conf=self.conf, iou=self.iou, verbose=False, imgsz=800)
        raw_structural = list(raw_combined_res)
        
        # ── Step 2: Placeholder for specialized insulator detector (Run on pole-top crop later) ──
        raw_insulator = None

        # ── Step 2.5: Run specialized crossarm detector (if available) ──
        raw_crossarm = None
        if self.crossarm_detector:
            print("DEBUG: Running crossarm detector...")
            raw_crossarm = self.crossarm_detector(image_path, conf=THRESHOLD_CROSSARM, imgsz=800, verbose=False)

        # Proactive memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── Step 3: Parse results into typed lists (Separate Streams) ────
        insulator_boxes  = []   # (box, conf, angle_deg)
        pole_boxes_raw   = []   # (box, conf, angle_deg)
        crossarm_boxes   = []   # (box, conf, angle_deg)
        conductor_boxes  = []   # (box, conf)
        street_light_boxes = [] # (box, conf, poly)
        other_boxes        = [] # (label, box, conf, poly)
        flags = defaultdict(bool)

        # Process structural model output
        total_structural = 0
        for result in raw_structural:
            obb   = result.obb   if hasattr(result, "obb")   and result.obb else None
            masks = result.masks if hasattr(result, "masks") and result.masks else None
            boxes = result.boxes if hasattr(result, "boxes") and result.boxes else None
            
            print(f"DEBUG: Found {len(masks) if masks else 0} masks, {len(obb) if obb else 0} obb, {len(boxes) if boxes else 0} boxes.")

            # ── Handle Segmentation Masks (New Model Support) ──
            if masks is not None and len(masks) > 0:
                for i in range(len(masks)):
                    # Masks object doesn't have cls/conf; they are in boxes
                    cls_name = self.component_model.names[int(boxes.cls[i])]
                    conf_val = float(boxes.conf[i])
                    print(f"DEBUG: Mask - Class: {cls_name}, Conf: {conf_val:.2f}")
                    total_structural += 1
                    
                    # Extract bbox from boxes (Masks object doesn't have xyxy)
                    b = boxes.xyxy[i].cpu().numpy()
                    box = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
                    angle_deg = None # Use aspect-ratio fallback for segmentation
                    
                    # Polygon for visualization
                    poly = [[int(b[0]), int(b[1])], [int(b[2]), int(b[1])], [int(b[2]), int(b[3])], [int(b[0]), int(b[3])]]
                    
                    if _match_keyword(cls_name, "pole") and conf_val >= THRESHOLD_POLE:
                        try:
                            from skimage.morphology import skeletonize
                            w, h = int(b[2] - b[0]), int(b[3] - b[1])
                            if w > 0 and h > 0:
                                pole_mask = np.zeros((h, w), dtype=np.uint8)
                                poly_pts = masks.xy[i].astype(np.int32)
                                if len(poly_pts) > 0:
                                    poly_pts[:, 0] -= int(b[0])
                                    poly_pts[:, 1] -= int(b[1])
                                    cv2.fillPoly(pole_mask, [poly_pts], 1)
                                    skeleton = skeletonize(pole_mask > 0)
                                    y_skel, x_skel = np.nonzero(skeleton)
                                    if len(x_skel) > 10:
                                        m, c_val = np.polyfit(y_skel, x_skel, 1)
                                        skel_angle = 90.0 - np.degrees(np.arctan(m))
                                        skel_deviation = abs(90.0 - abs(skel_angle))
                                        print(f"DEBUG: Skeleton lean angle calculated: {skel_angle:.2f}° (deviation: {np.degrees(np.arctan(m)):.2f}°)")
                                        
                                        # Sanity check: reject main_pole skeletons deviating > 35° from vertical
                                        is_strut_cls = ("strut" in cls_name.lower())
                                        if not is_strut_cls and skel_deviation > 35.0:
                                            print(f"🚫 [Skeleton Filter] Rejected {cls_name} (conf={conf_val:.2f}): skeleton {skel_angle:.1f}° deviates {skel_deviation:.1f}° from vertical")
                                            angle_deg = None  # Fall back to AR-based classification
                                        else:
                                            angle_deg = skel_angle
                        except Exception as e:
                            print(f"DEBUG: Skeletonization failed: {e}")

                    if _match_keyword(cls_name, "conductor") and conf_val >= THRESHOLD_CONDUCTOR:
                        conductor_boxes.append((box, conf_val, poly))
                    elif _match_keyword(cls_name, "pole") and conf_val < THRESHOLD_POLE:
                        # Skip low-confidence poles (below 80%)
                        continue
                    else:
                        is_strut = ("strut" in cls_name.lower())
                        self._categorise(cls_name, box, conf_val, angle_deg, insulator_boxes, pole_boxes_raw, crossarm_boxes, conductor_boxes, street_light_boxes, other_boxes, flags, polygon=poly, is_strut=is_strut)

            # ── Handle OBB (Old Model Support) ──
            if obb is not None and len(obb) > 0:
                for i in range(len(obb)):
                    cls_name  = self.component_model.names[int(obb.cls[i])]
                    conf_val  = float(obb.conf[i])
                    total_structural += 1
                    
                    # Extract rotated angle for crossarm/pole classification
                    xywhr = obb.xywhr[i].cpu().numpy()
                    cx, cy, bw, bh, angle_rad = xywhr
                    angle_deg = math.degrees(float(angle_rad))

                    # ── Calculate the 4 rotated points (Polygon) ───────
                    cos_a = math.cos(float(angle_rad))
                    sin_a = math.sin(float(angle_rad))
                    dx, dy = bw / 2, bh / 2
                    
                    # Rotated corners relative to center
                    p1 = [cx + (-dx*cos_a - -dy*sin_a), cy + (-dx*sin_a + -dy*cos_a)]
                    p2 = [cx + ( dx*cos_a - -dy*sin_a), cy + ( dx*sin_a + -dy*cos_a)]
                    p3 = [cx + ( dx*cos_a -  dy*sin_a), cy + ( dx*sin_a +  dy*cos_a)]
                    p4 = [cx + (-dx*cos_a -  dy*sin_a), cy + (-dx*sin_a +  dy*cos_a)]
                    poly = [[int(pt[0]), int(pt[1])] for pt in [p1, p2, p3, p4]]

                    # Use native AABB (Axis-Aligned Bounding Box) for consistency with training
                    b = obb.xyxy[i].cpu().numpy()
                    box = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
                    
                    # ── Class-Specific Sensitivity Filtering ──
                    if _match_keyword(cls_name, "conductor") and conf_val >= THRESHOLD_CONDUCTOR:
                        conductor_boxes.append((box, conf_val, poly))
                    elif _match_keyword(cls_name, "insulator") and conf_val >= THRESHOLD_INSULATOR:
                         self._categorise(cls_name, box, conf_val, angle_deg, insulator_boxes, pole_boxes_raw, crossarm_boxes, conductor_boxes, street_light_boxes, other_boxes, flags, polygon=poly)
                    elif _match_keyword(cls_name, "pole") and conf_val >= THRESHOLD_POLE:
                         is_strut = ("strut" in cls_name.lower())
                         self._categorise(cls_name, box, conf_val, angle_deg, insulator_boxes, pole_boxes_raw, crossarm_boxes, conductor_boxes, street_light_boxes, other_boxes, flags, polygon=poly, is_strut=is_strut)
                    elif _match_keyword(cls_name, "crossarm") and conf_val >= THRESHOLD_CROSSARM:
                         self._categorise(cls_name, box, conf_val, angle_deg, insulator_boxes, pole_boxes_raw, crossarm_boxes, conductor_boxes, street_light_boxes, other_boxes, flags, polygon=poly)
                    elif conf_val >= 0.40: # Lowered threshold to catch more "other" objects (DTR, lattice, vegetation, etc.)
                         is_strut = ("strut" in cls_name.lower())
                         self._categorise(cls_name, box, conf_val, angle_deg, insulator_boxes, pole_boxes_raw, crossarm_boxes, conductor_boxes, street_light_boxes, other_boxes, flags, polygon=poly, is_strut=is_strut)

            # Only process raw boxes if NO masks were found (avoid duplicate processing)
            if (masks is None or len(masks) == 0) and boxes is not None and len(boxes) > 0:
                for box_obj in boxes:
                    cls_name = self.component_model.names[int(box_obj.cls)]
                    conf_val = float(box_obj.conf)
                    total_structural += 1
                    
                    b        = box_obj.xyxy[0].cpu().numpy()
                    box      = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
                    bw, bh   = b[2] - b[0], b[3] - b[1]
                    angle_deg = None # Trigger aspect-ratio fallback
                    
                    # For non-OBB detections, the polygon is just the bbox corners
                    poly = [[int(b[0]), int(b[1])], [int(b[2]), int(b[1])], [int(b[2]), int(b[3])], [int(b[0]), int(b[3])]]

                    # ── Class-Specific Sensitivity Filtering ──
                    if _match_keyword(cls_name, "conductor") and conf_val >= THRESHOLD_CONDUCTOR:
                        conductor_boxes.append((box, conf_val, poly))
                    elif _match_keyword(cls_name, "insulator") and conf_val >= THRESHOLD_INSULATOR:
                         self._categorise(cls_name, box, conf_val, angle_deg, insulator_boxes, pole_boxes_raw, crossarm_boxes, conductor_boxes, street_light_boxes, other_boxes, flags, polygon=poly)
                    elif _match_keyword(cls_name, "pole") and conf_val >= THRESHOLD_POLE:
                         # 1. Trust model's explicit class first
                         is_strut = ("strut" in cls_name.lower())
                         if is_strut:
                             # If model says it's a strut, BYPASS all other filters
                             self._categorise(cls_name, box, conf_val, angle_deg, insulator_boxes, pole_boxes_raw, crossarm_boxes, conductor_boxes, street_light_boxes, other_boxes, flags, polygon=poly, is_strut=True)
                             continue

                         # 2. Geometric backup if model just says 'pole'
                         if angle_deg is not None:
                             lean = abs(angle_deg - 90)
                             if lean > 15.0: is_strut = True
                         
                         self._categorise(cls_name, box, conf_val, angle_deg, insulator_boxes, pole_boxes_raw, crossarm_boxes, conductor_boxes, street_light_boxes, other_boxes, flags, polygon=poly, is_strut=is_strut)
        
        # ── Process Crossarm Model Output ──
        if raw_crossarm is not None:
            raw_crossarm_list = list(raw_crossarm)
            print(f"DEBUG: Processing {len(raw_crossarm_list)} crossarm results")
            for result_ca in raw_crossarm_list:
                boxes = result_ca.boxes if hasattr(result_ca, "boxes") and result_ca.boxes else None
                print(f"DEBUG: Crossarm boxes found: {len(boxes) if boxes is not None else 0}")
                if boxes is not None and len(boxes) > 0:
                    for box_obj in boxes:
                        cls_name = self.crossarm_detector.names[int(box_obj.cls)]
                        
                        # Support mapping for best_components.pt actual output classes to pretty UI names
                        best_map = {
                            "V_Cross_Arm": "V Cross Arm",
                            "Tapping_Arm": "Tapping Arm",
                            "Side_Arm": "Side Arm",
                            "T_rising": "T-rising",
                            "Top_Cleat": "Top Cleat",
                            "Special_Clamp": "Special Clamp",
                            "Stay_Set": "Stay Set",
                            "AB_Switch": "AB Switch",
                            "Street_Light": "Street Light",
                            "Box_Arm": "Box Arm",
                            "DTR": "DTR",
                        }
                        if cls_name in best_map:
                            cls_name = best_map[cls_name]

                        # Skip insulators and poles from this model as requested:
                        # "better for all detections except insulators and poles"
                        if _match_keyword(cls_name, "insulator") or _match_keyword(cls_name, "pole"):
                            continue
                            
                        conf_val = float(box_obj.conf)
                        print(f"DEBUG: Crossarm model detected '{cls_name}' with conf={conf_val:.2f}")
                        if conf_val >= THRESHOLD_CROSSARM:
                            b   = box_obj.xyxy[0].cpu().numpy()
                            box = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
                            poly = [[int(b[0]), int(b[1])], [int(b[2]), int(b[1])], [int(b[2]), int(b[3])], [int(b[0]), int(b[3])]]
                            self._categorise(cls_name, box, conf_val, None, insulator_boxes, pole_boxes_raw, crossarm_boxes, conductor_boxes, street_light_boxes, other_boxes, flags, polygon=poly)
                        else:
                            print(f"DEBUG: Crossarm '{cls_name}' rejected (conf {conf_val:.2f} < {THRESHOLD_CROSSARM})")
        
        # ── Step 2.7: Generate Pole-Top Crop ──
        crop_x1, crop_y1, crop_x2, crop_y2 = self._generate_pole_top_crop(img_original, crossarm_boxes, pole_boxes_raw)
        crop_img = img_original[crop_y1:crop_y2, crop_x1:crop_x2]
        
        # ── Step 2.8: Run 4-class insulator detector on Crop ──
        print("DEBUG: Running 4-class insulator detector on pole-top crop...")
        if crop_img.size > 0:
            raw_insulator = self.insulator_detector(crop_img, conf=THRESHOLD_INSULATOR, imgsz=800, verbose=False)
        else:
            raw_insulator = []

        # Process specialized insulator detector output (Final Insulator Detections)
        for result in raw_insulator:
            boxes = result.boxes if hasattr(result, "boxes") and result.boxes else None
            masks = result.masks if hasattr(result, "masks") and result.masks else None
            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    box_obj = boxes[i]
                    cls_name = self.insulator_detector.names[int(box_obj.cls)]
                    conf_val = float(box_obj.conf)
                    b        = box_obj.xyxy[0].cpu().numpy()
                    
                    # Map coordinates back to the original image space
                    x1 = int(b[0]) + crop_x1
                    y1 = int(b[1]) + crop_y1
                    x2 = int(b[2]) + crop_x1
                    y2 = int(b[3]) + crop_y1
                    box = (x1, y1, x2, y2)
                    angle_deg = None

                    # Extract polygon points from mask if available
                    poly = None
                    if masks is not None and masks.xy is not None and i < len(masks.xy):
                        poly_pts = masks.xy[i]
                        if len(poly_pts) > 0:
                            poly = [[int(pt[0] + crop_x1), int(pt[1] + crop_y1)] for pt in poly_pts]
                            
                    if poly is None:
                        poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    
                    if _match_keyword(cls_name, "conductor") and conf_val >= 0.05:
                        conductor_boxes.append((box, conf_val, poly))
                    else:
                        self._categorise(
                            cls_name, box, conf_val, angle_deg,
                            insulator_boxes, pole_boxes_raw,
                            crossarm_boxes, conductor_boxes,
                            street_light_boxes, other_boxes,
                            flags, polygon=poly
                        )
        
        # ── Deduplicate multi-scale detections (NMS) ─────────
        insulator_boxes = self._nms(insulator_boxes, iou_threshold=0.75) 
        pole_boxes_raw  = self._nms(pole_boxes_raw,  iou_threshold=0.45)
        crossarm_boxes  = self._nms(crossarm_boxes,  iou_threshold=0.45)
        conductor_boxes = self._nms(conductor_boxes, iou_threshold=0.65)
        street_light_boxes = self._nms(street_light_boxes, iou_threshold=0.45)
        other_boxes        = self._nms(other_boxes,        iou_threshold=0.45)

        # ── Infer missing pole logic ──────────────
        # DISABLED for strict 80% confidence requirement
        # if not pole_boxes_raw:
        #     inferred = self._infer_pole_if_missing(insulator_boxes, crossarm_boxes, img_h, img_w)
        #     ...

        # ── Calculate Global Tilt Compensation ────────────────
        # If camera is tilted, all straight components will share the same offset.
        tilt_samples = []
        for _, _, p_angle, _, _ in pole_boxes_raw:
            if p_angle is not None:
                # Pole ideal is 90
                offset = p_angle - 90
                # We normalize offset to [-45, 45]
                if offset > 90: offset -= 180
                if offset < -90: offset += 180
                if abs(offset) < GLOBAL_TILT_MAX_DEG:
                    tilt_samples.append(offset)
        
        for _, _, c_angle, _, _ in crossarm_boxes:
            if c_angle is not None:
                # Crossarm ideal is 0 (or 180)
                offset = c_angle
                if offset > 90: offset -= 180
                if offset < -90: offset += 180
                if abs(offset) < GLOBAL_TILT_MAX_DEG:
                    tilt_samples.append(offset)
        
        global_tilt = 0.0
        if tilt_samples:
            import statistics
            global_tilt = statistics.median(tilt_samples)
            flags["tilt_compensated"] = True
            flags["global_tilt_deg"] = round(global_tilt, 1)

        # ── Classify each insulator ───────────────────────────
        insulator_results = []
        for box, conf_val, angle_deg, polygon, detected_cls in insulator_boxes:
            ins_result = self.insulator_clf.classify(
                img_original, box, conf_val, obb_angle_deg=angle_deg, detected_class=detected_cls
            )
            ins_result.obb_polygon = polygon
            insulator_results.append(ins_result)

        all_poles = []
        pole_result = None
        if pole_boxes_raw:
            # Sort: main_poles first (by confidence), then strut_poles
            # This ensures the primary pole is always the most confident main_pole
            pole_boxes_raw.sort(
                key=lambda x: (1 if x[4] else 0, -x[1]),  # main_pole(0) before strut(1), then by -confidence
            )
            
            print(f"DEBUG: Pole sort order: {[(('STRUT' if x[4] else 'MAIN'), f'conf={x[1]:.2f}', f'angle={x[2]}') for x in pole_boxes_raw]}")
            
            for i, (p_box, p_conf, p_angle, p_poly, p_is_strut) in enumerate(pole_boxes_raw):
                pr = classify_pole_orientation(p_box, p_angle, is_strut=p_is_strut, tilt_compensation=global_tilt)
                pr.detection_conf = p_conf
                pr.obb_polygon = p_poly
                
                all_poles.append(pr)
                
                # The first pole (highest priority main_pole) is the primary one
                if i == 0:
                    pole_result = pr
                    print(f"DEBUG: Primary pole selected: type={pr.pole_type}, lean={pr.lean_angle_deg:.1f}°, conf={p_conf:.2f}")

        # ── Run OCR on all detected poles ─────────────────────
        pole_id = "Not Found"
        crossarm_results = []
        for box, conf, ang, poly, native_cls in crossarm_boxes:
            cr = classify_crossarm_shape(
                box,
                [c[0] for c in conductor_boxes],
                [p[0] for p in pole_boxes_raw],
                (img_h, img_w),
                obb_angle_deg=ang,
                insulator_results=insulator_results,
                native_class=native_cls,
                tilt_compensation=global_tilt
            )
            cr.detection_conf = conf
            cr.obb_polygon = poly
            crossarm_results.append(cr)

        dominant_shape, n_crossarms, crossarm_faults = aggregate_crossarm_results(crossarm_results)

        # ── Collect all adjustment faults ─────────────────────
        adjustment_faults = []

        for ins in insulator_results:
            if ins.adjustment_fault:
                adjustment_faults.append({
                    "component": f"insulator ({ins.type_final})",
                    "severity":  ins.fault_severity,
                    "note":      ins.fault_note,
                })

        if pole_result and pole_result.adjustment_fault:
            adjustment_faults.append({
                "component": "pole",
                "severity":  pole_result.fault_severity,
                "note":      pole_result.fault_note,
            })

        for cr in crossarm_faults:
            adjustment_faults.append({
                "component": f"crossarm ({cr.shape})",
                "severity":  cr.fault_severity,
                "note":      cr.fault_note,
            })

        # ── Detect HT + LT on same pole ──────────────────────
        if conductor_boxes and len(conductor_boxes) >= 4:
            y_centres = [(det[0][1] + det[0][3]) / 2 for det in conductor_boxes]
            y_range = max(y_centres) - min(y_centres)
            
            if y_range > img_h * HT_LT_HEIGHT_THRESHOLD:
                flags["has_ht_and_lt"] = True
                
        if flags.get("has_ht_and_lt"):
            voltages = {i.voltage for i in insulator_results}
            if len(voltages) == 1 and list(voltages)[0] in ["11kV", "33kV"]:
                flags["has_ht_and_lt"] = False

        # ── Populate signals for Rule Engine ──────────────────
        max_v = "unknown"
        max_s = 0
        max_c = "low"
        
        all_voltages = [i.voltage for i in insulator_results if i.voltage != "unknown"]
        v_priority = {"33kV": 4, "11kV": 3, "6.3kV": 2, "LT": 1}
        if all_voltages:
            max_v = max(all_voltages, key=lambda v: v_priority.get(v, 0))
            max_c = "high" if any(i.type_confidence == "high" for i in insulator_results) else "medium"
            max_s = max([i.shed_count for i in insulator_results], default=0)
            
        if any(i.type_final == "pin" for i in insulator_results):
            ins_type_signal = "pin"
        elif any(i.type_final == "disc" for i in insulator_results):
            ins_type_signal = "disc"
        elif any(i.type_final == "shackle" for i in insulator_results):
            ins_type_signal = "shackle"
        else:
            ins_type_signal = "unknown"

        signals = ComponentSignals(
            insulator_type    = ins_type_signal,
            insulator_voltage = max_v,
            shed_count        = max_s,
            insulator_conf    = max_c,

            has_dtr      = flags["has_dtr"],
            has_ab_cable = flags["has_ab_cable"],
            has_lattice  = flags["has_lattice"],
            has_jumper   = flags["has_jumper"],
            has_ht_and_lt= flags["has_ht_and_lt"],
            has_broken_wire = flags["has_broken_wire"],
            has_vegetation  = flags["has_vegetation"],

            pole_type      = pole_result.pole_type      if pole_result else "vertical_pole",
            lean_angle_deg = pole_result.lean_angle_deg if pole_result else 0.0,

            crossarm_count  = n_crossarms,
            crossarm_shape  = dominant_shape,
            conductor_count = len(conductor_boxes),
        )

        # ── Rule engine → final class ─────────────────────────
        classification = classify_pole(signals)

        # ── Build result ──────────────────────────────────────
        pipeline_result = PipelineResult(
            final_class      = classification.final_class,
            class_id         = classification.class_id,
            reason           = classification.reason,
            voltage          = classification.voltage,
            confidence       = classification.confidence,
            signals_used     = classification.signals_used,
            insulators       = insulator_results,
            all_poles        = all_poles,
            pole_orientation = pole_result,
            crossarms        = crossarm_results,
            conductors       = conductor_boxes,
            street_lights    = street_light_boxes,
            others           = other_boxes,
            crossarm_shape   = dominant_shape,
            crossarm_count   = n_crossarms,
            conductor_count  = len(conductor_boxes),
            flags            = dict(flags),
            adjustment_faults= adjustment_faults,
        )

        # ── Visualize ─────────────────────────────────────────
        if visualize:
            # Include detections in visualization
            raw_vis_sources = raw_structural + list(raw_insulator)
            vis = self._draw(img, pipeline_result, raw_vis_sources)
            out = save_path or (Path(image_path).stem + "_result.jpg")
            cv2.imwrite(str(out), vis)
            print(f"Saved: {out}")

        return pipeline_result

    def _generate_pole_top_crop(
        self,
        img: np.ndarray,
        crossarm_boxes: list,
        pole_boxes: list,
    ) -> Tuple[int, int, int, int]:
        """
        Generates a crop box (x1, y1, x2, y2) around the pole-top structure.
        """
        img_h, img_w = img.shape[:2]
        
        # 1. Primary: Use detected crossarm structures
        if crossarm_boxes:
            # Each entry in crossarm_boxes is: (box, conf, angle, polygon, native)
            x1_min = min(c[0][0] for c in crossarm_boxes)
            y1_min = min(c[0][1] for c in crossarm_boxes)
            x2_max = max(c[0][2] for c in crossarm_boxes)
            y2_max = max(c[0][3] for c in crossarm_boxes)
            
            cw = x2_max - x1_min
            ch = y2_max - y1_min
            
            # Apply padding around crossarm structure
            pad_x = max(60, int(cw * 0.15))
            pad_y_top = max(150, int(cw * 0.25))     # more vertical padding on top for pins
            pad_y_bottom = max(150, int(cw * 0.20))  # vertical padding below for discs/shackles
            
            crop_x1 = max(0, x1_min - pad_x)
            crop_y1 = max(0, y1_min - pad_y_top)
            crop_x2 = min(img_w, x2_max + pad_x)
            crop_y2 = min(img_h, y2_max + pad_y_bottom)
            
            if (crop_x2 - crop_x1) > 20 and (crop_y2 - crop_y1) > 20:
                print(f"DEBUG: Pole-top crop generated from crossarms: x1={crop_x1}, y1={crop_y1}, x2={crop_x2}, y2={crop_y2}")
                return crop_x1, crop_y1, crop_x2, crop_y2

        # 2. Fallback: If no crossarms, crop the top 45% of the primary pole
        if pole_boxes:
            # Each entry in pole_boxes is: (box, conf, angle, polygon, is_strut)
            sorted_poles = sorted(pole_boxes, key=lambda x: x[1], reverse=True)
            p_box = sorted_poles[0][0]
            px1, py1, px2, py2 = p_box
            ph = py2 - py1
            pw = px2 - px1
            
            top_y2 = py1 + int(ph * 0.45)
            pad_x = max(200, int(pw * 3.0))
            
            crop_x1 = max(0, px1 - pad_x)
            crop_y1 = max(0, py1 - 50)
            crop_x2 = min(img_w, px2 + pad_x)
            crop_y2 = min(img_h, top_y2 + 100)
            
            if (crop_x2 - crop_x1) > 20 and (crop_y2 - crop_y1) > 20:
                print(f"DEBUG: Pole-top crop generated from pole fallback: x1={crop_x1}, y1={crop_y1}, x2={crop_x2}, y2={crop_y2}")
                return crop_x1, crop_y1, crop_x2, crop_y2
                
        # 3. Final Fallback: Entire image
        print("DEBUG: No crossarms or poles detected. Falling back to full image crop.")
        return 0, 0, img_w, img_h

    # ── Internal helpers ──────────────────────────────────────

    def _dominant_insulator(self, results: List[InsulatorResult]) -> Optional[InsulatorResult]:
        """Finds the most significant (highest voltage) insulator detected."""
        if not results:
            return None
        
        v_priority = {"33kV": 4, "11kV": 3, "6.3kV": 2, "LT": 1, "unknown": 0}
        
        # Sort by voltage priority, then confidence
        ranked = sorted(
            results,
            key=lambda x: (v_priority.get(x.voltage, 0), 1 if x.type_confidence == "high" else 0),
            reverse=True
        )
        return ranked[0]

    def _categorise(
        self, cls_name, box, conf_val, angle_deg,
        insulator_boxes, pole_boxes_raw,
        crossarm_boxes, conductor_boxes,
        street_light_boxes, other_boxes, flags,
        polygon=None, is_strut=False
    ):
        """Routes a detection into the right typed list, including polygon data."""
        det = (box, conf_val, angle_deg, polygon, is_strut)
        if _match_keyword(cls_name, "insulator"):
            det = (box, conf_val, angle_deg, polygon, cls_name)
            insulator_boxes.append(det)
        elif _match_keyword(cls_name, "pole"):
            # GEOMETRY SHIELD: Real main poles are tall and thin.
            # (Strut poles are exempt because their lean makes their AABB wide)
            w, h = box[2] - box[0], box[3] - box[1]
            if not is_strut and h < (w * 0.5):
                # Reject purely horizontal structures like walls/buildings
                print(f"🚫 [Geometry Shield] Rejected wide detection: {cls_name} (Aspect Ratio: {h/w:.1f})")
                return
            pole_boxes_raw.append(det)
        elif _match_keyword(cls_name, "crossarm"):
            native = cls_name.lower()
            # Structural Guard: Only use geometric check if the model didn't 
            # already tell us it's a pole.
            w, h = box[2] - box[0], box[3] - box[1]
            if (w * 1.2) > h or "t_rising" in native or "t-rising" in native or "side_arm" in native or "arm" in native:
                crossarm_boxes.append((box, conf_val, angle_deg, polygon, native))
            else:
                # If it's too tall for a crossarm, it's likely a pole fragment
                # that the model mislabeled as a crossarm.
                is_strut = ("strut" in native)
                pole_boxes_raw.append((box, conf_val, angle_deg, polygon, is_strut))
        elif _match_keyword(cls_name, "conductor"):
            conductor_boxes.append((box, conf_val, polygon))
        elif _match_keyword(cls_name, "lamp_head"):
            street_light_boxes.append((box, conf_val, polygon))
        elif _match_keyword(cls_name, "dtr_tank"):
            flags["has_dtr"] = True
            other_boxes.append(("DTR Tank", box, conf_val, polygon))
        elif _match_keyword(cls_name, "ab_cable"):
            flags["has_ab_cable"] = True
            other_boxes.append(("AB Cable", box, conf_val, polygon))
        elif _match_keyword(cls_name, "lattice"):
            flags["has_lattice"] = True
            other_boxes.append(("Lattice Frame", box, conf_val, polygon))
        elif _match_keyword(cls_name, "jumper"):
            flags["has_jumper"] = True
            other_boxes.append(("Jumper Wire", box, conf_val, polygon))
        elif _match_keyword(cls_name, "broken_wire"):
            flags["has_broken_wire"] = True
            other_boxes.append(("WIRE_BROKEN", box, conf_val, polygon))
        elif _match_keyword(cls_name, "vegetation"):
            flags["has_vegetation"] = True
            other_boxes.append(("VEGETATION", box, conf_val, polygon))
        else:
            # Catch-all for any other model classes not explicitly handled
            other_boxes.append((cls_name.upper(), box, conf_val, polygon))

    def _infer_pole_if_missing(
        self, insulator_boxes, crossarm_boxes, img_h, img_w
    ) -> Optional[Tuple[Tuple[int, int, int, int], float, float]]:
        """
        If we see insulators/crossarms but the model missed the pole (e.g. cut off),
        infer a vertical pole box from the component alignment.
        """
        all_box_coords = []
        for item in insulator_boxes: all_box_coords.append(item[0])
        for item in crossarm_boxes:  all_box_coords.append(item[0])
        
        if not all_box_coords:
            return None
            
        # Composite bounding box of all components
        x1_min = min(b[0] for b in all_box_coords)
        y1_min = min(b[1] for b in all_box_coords)
        x2_max = max(b[2] for b in all_box_coords)
        y2_max = max(b[3] for b in all_box_coords)
        
        avg_cx = (x1_min + x2_max) / 2
        
        # Create a vertical pole box (narrow and tall)
        # We extend it to the bottom of the image because it's usually a vertical support
        pole_width = max(60, int((x2_max - x1_min) * 0.4))
        inf_x1 = int(max(0, avg_cx - pole_width // 2))
        inf_x2 = int(min(img_w, avg_cx + pole_width // 2))
        inf_y1 = int(max(0, y1_min - 20))
        inf_y2 = img_h # Extend to bottom
        
        # Return as a low-confidence detection
        return ((inf_x1, inf_y1, inf_x2, inf_y2), 0.50, 90.0)

    def _calculate_max_overlap(self, box1: tuple, box2: tuple) -> float:
        """Computes Intersection-over-Minimum-Area (IoM). Better for nested boxes."""
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        
        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area  = (x2 - x1) * (y2 - y1)
        box2_area  = (x4 - x3) * (y4 - y3)
        
        min_area = min(box1_area, box2_area)
        if min_area <= 0: return 0.0
        
        return inter_area / min_area

    def _nms(self, items: list, iou_threshold: float = 0.45) -> list:
        """
        Simple NMS for list of detections using IoM overlap.
        """
        if not items:
            return []
            
        # Helper to extract (box, score, class, original_item)
        def parse_item(item):
            if isinstance(item[0], str):  # e.g., other_boxes: (label, box, conf, poly)
                return item[1], item[2], item[0], item
            else:  # e.g., (box, conf, ...)
                box = item[0]
                score = item[1]
                cls = item[4] if len(item) >= 5 else None
                return box, score, cls, item

        parsed = [parse_item(item) for item in items]
        
        # Sort parsed items by score descending
        parsed_sorted = sorted(parsed, key=lambda x: x[1], reverse=True)
        keep_parsed = []
        
        while parsed_sorted:
            best = parsed_sorted.pop(0)
            keep_parsed.append(best)
            
            remaining = []
            for item in parsed_sorted:
                # Class-aware suppression (if classes are specified and different)
                if best[2] is not None and item[2] is not None and best[2] != item[2]:
                    # Exception 1: If they are both poles (represented by is_strut flag being boolean),
                    # and they heavily overlap (IoM >= 0.65), it's the same physical pole.
                    # In this case, we suppress the lower-confidence duplicate regardless of class.
                    is_best_pole = isinstance(best[2], bool)
                    is_item_pole = isinstance(item[2], bool)
                    if is_best_pole and is_item_pole:
                        overlap = self._calculate_max_overlap(best[0], item[0])
                        if overlap >= 0.65:
                            # Suppress: Do not add to remaining
                            continue
                            
                    # Exception 2: If they are both string classes (insulators, crossarms, etc.)
                    # and they heavily overlap (IoM >= 0.50), it's the same physical component.
                    # We suppress the lower-confidence duplicate regardless of class.
                    is_best_str = isinstance(best[2], str)
                    is_item_str = isinstance(item[2], str)
                    if is_best_str and is_item_str:
                        overlap = self._calculate_max_overlap(best[0], item[0])
                        if overlap >= 0.50:
                            # Suppress: Do not add to remaining
                            continue
                            
                    remaining.append(item)
                    continue

                overlap = self._calculate_max_overlap(best[0], item[0])
                # Relaxed threshold for struts (allow them to be close together)
                # best[2] is is_strut for pole boxes
                is_strut = best[2] if isinstance(best[2], bool) else False
                current_threshold = 0.85 if is_strut else iou_threshold
                if overlap < current_threshold:
                    remaining.append(item)
            parsed_sorted = remaining
            
        # Reconstruct the filtered list in their original formats
        keep = [p[3] for p in keep_parsed]
        
        # 2. Vertical Merge (Fixes fragmented poles)
        # Only apply to pole_boxes_raw (which are 5-tuples and have is_strut flag at index 4)
        is_pole_list = len(keep) > 0 and len(keep[0]) == 5 and isinstance(keep[0][4], bool)
        if not is_pole_list:
            return keep
            
        merged = []
        keep = sorted(keep, key=lambda x: x[0][1]) # Sort by Y1 (top to bottom)
        
        while keep:
            curr = keep.pop(0)
            c_box, c_conf, c_angle, c_poly, c_is_strut = curr
            
            # Look for a fragment directly below this one
            found_fragment = False
            for i, next_item in enumerate(keep):
                n_box, n_conf, n_angle, n_poly, n_is_strut = next_item
                
                # Check if they are the same type and vertically close
                if c_is_strut == n_is_strut:
                    # Check X-overlap (Vertical Alignment)
                    x_overlap = min(c_box[2], n_box[2]) - max(c_box[0], n_box[0])
                    
                    # If they share > 50% width and the gap is small
                    gap = n_box[1] - c_box[3]
                    if x_overlap / max(1, (c_box[2]-c_box[0])) > 0.5 and gap < 100:
                        # MERGE!
                        new_box = (
                            min(c_box[0], n_box[0]), 
                            min(c_box[1], n_box[1]),
                            max(c_box[2], n_box[2]),
                            max(c_box[3], n_box[3])
                        )
                        # Average angle/confidence
                        avg_conf = (c_conf + n_conf) / 2
                        avg_angle = None
                        if c_angle is not None and n_angle is not None:
                            avg_angle = (c_angle + n_angle) / 2
                        elif c_angle is not None:
                            avg_angle = c_angle
                        else:
                            avg_angle = n_angle
                            
                        new_poly = c_poly + n_poly
                        
                        # Update current and remove next_item from keep
                        curr = (new_box, avg_conf, avg_angle, new_poly, c_is_strut)
                        keep.pop(i)
                        found_fragment = True
                        break
            
            merged.append(curr)
            if found_fragment:
                # Sort again to preserve top-to-bottom order for subsequent merges
                keep = sorted(keep, key=lambda x: x[0][1])
                
        return merged

    def _enhance_image(self, img: np.ndarray) -> np.ndarray:
        """
        Applies CLAHE, Gamma Correction, and light Sharpening to help with
        low-contrast areas and "silhouette" poles.
        """
        # 1. CLAHE (Local Contrast)
        # -------------------------
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        img_clahe = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # 2. Gamma Correction (Shadow recovery)
        # -------------------------------------
        # Gamma > 1.0 brightens shadows (nonlinear)
        gamma = 1.5
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 
                          for i in np.arange(0, 256)]).astype("uint8")
        img_gamma = cv2.LUT(img_clahe, table)

        # 3. Light Sharpening (Edge definition)
        # -------------------------------------
        # Unsharp masking approach: sharpened = original + (original - blurred)*amount
        gaussian_blur = cv2.GaussianBlur(img_gamma, (0, 0), 3)
        img_sharpened = cv2.addWeighted(img_gamma, 1.5, gaussian_blur, -0.5, 0)

        return img_sharpened

    def _dominant_insulator(
        self,
        results: list,
    ) -> Optional[InsulatorResult]:
        """
        Consensus-based signal aggregation.
        If multiple insulators are detected on a single pole, we use majority voting 
        for type and voltage to avoid noisy single-object errors (e.g. miscounting sheds 
        on one insulator).
        """
        if not results:
            return None
            
        from collections import Counter
        
        # 1. Consensus on Type (Pin vs Disc)
        types = [r.type_final for r in results]
        dominant_type = Counter(types).most_common(1)[0][0]
        
        # 2. Consensus on Voltage (11kV vs 33kV)
        # Filter unknown for cleaner signal
        valid_vs = [r.voltage for r in results if r.voltage != "unknown"]
        dominant_voltage = Counter(valid_vs).most_common(1)[0][0] if valid_vs else "unknown"
        
        # 3. Choose a representative result that matches the consensus
        for r in results:
            if r.type_final == dominant_type and r.voltage == dominant_voltage:
                return r

        return results[0]

    def _draw(
        self,
        img: np.ndarray,
        result: PipelineResult,
        raw_detections,
    ) -> np.ndarray:
        """Draws detections + classification banner on image."""
        vis   = img.copy()
        img_h, img_w = img.shape[:2]

        # Draw raw detections (Standard + OBB)
        for r in raw_detections:
            if hasattr(r, 'boxes') and r.boxes is not None:
                for box in r.boxes:
                    b = box.xyxy[0].cpu().numpy().astype(int)
                    cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (180, 180, 180), 1)
            if hasattr(r, 'obb') and r.obb is not None:
                for obb in r.obb:
                    b = obb.xyxy[0].cpu().numpy().astype(int)
                    cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (140, 140, 140), 1)

        # Highlight insulator detections with type + shed count + CONFIDENCE
        for ins in result.insulators:
            x1, y1, x2, y2 = ins.box
            color = (50, 220, 50) if ins.type_final == "pin" else (50, 50, 220)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
            label = (
                f"{ins.type_final} ({ins.detection_conf:.2f}) | "
                f"sheds={ins.shed_count} | "
                f"{ins.voltage}"
            )
            cv2.putText(vis, label, (x1, min(img_h - 6, y2 + 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)

        # Highlight classified crossarms + CONFIDENCE
        for ca in result.crossarms:
            x1, y1, x2, y2 = ca.box
            color = (255, 0, 255) # Magenta for crossarms
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"ARM ({ca.detection_conf:.2f}) | {ca.shape}"
            cv2.putText(vis, label, (x1, max(15, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Highlight final conductors in CYAN
        for det in result.conductors:
            box, conf = det[0], det[1]
            x1, y1, x2, y2 = box
            color = (255, 255, 0) # Cyan/Yellow-Cyan for visibility
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, f"wire:{conf:.2f}", (x1, max(15, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        # Highlight ALL detected poles with orientation labels
        for po in result.all_poles:
            x1, y1, x2, y2 = po.box
            is_inferred = result.flags.get("inferred_pole", False)
            
            # Label based on exact class types
            p_type = "MAIN POLE"
            if po.pole_type == "strut_pole":
                p_type = "STRUT POLE"

            color = (255, 255, 255) if not is_inferred else (200, 200, 200)  # White for main pole
            if po.pole_type == "strut_pole":
                color = (80, 127, 255)  # Coral/salmon for strut poles
            
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            # Display angle info: LEAN for main poles, ANGLE for strut poles
            if po.pole_type == "strut_pole":
                label = f"{p_type} | ANGLE: {po.lean_angle_deg:.1f}\u00b0"
            else:
                label = f"{p_type} | LEAN: {po.lean_angle_deg:.1f}\u00b0"
            if is_inferred: label = f"[INFERRED] {label}"
            cv2.putText(vis, label,
                        (x1, min(img_h - 6, y2 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Final class banner
        banner_h = 75
        banner = np.zeros((banner_h, img_w, 3), dtype=np.uint8)
        banner[:] = (25, 25, 25)
        conf_color = {
            "high": (50, 220, 50), "medium": (50, 180, 220), "low": (80, 80, 220)
        }.get(result.confidence, (150, 150, 150))

        cv2.putText(banner, f"CLASS: {result.final_class} ({result.confidence})",
                    (10, 26), cv2.FONT_HERSHEY_DUPLEX, 0.85, conf_color, 2)
        cv2.putText(banner, f"Reason: {result.reason}",
                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 170, 170), 1)
        cv2.putText(banner,
                    f"voltage={result.voltage} | "
                    f"crossarm={result.crossarm_shape} | "
                    f"wires={result.conductor_count} | "
                    f"POLE_ID: {result.pole_id}",
                    (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (120, 120, 120), 1)

        return np.vstack([banner, vis])


# ── CLI ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    
    # ── Configuration: Default Model Paths ────────────────────
    # These will be used if no command line arguments are provided.
    # Paths are relative to the project root (D:\NEW_ASAKTA\dry).
    ROOT = Path(__file__).parent.absolute()
    DEFAULT_COMP_MODEL = str(ROOT / "models" / "best_components.pt")
    DEFAULT_INS_MODEL  = str(ROOT / "models" / "insulator_new.pt")
    DEFAULT_SHED_MODEL = str(ROOT / "models" / "shed_model.pt")

    if len(sys.argv) < 2:
        print("\nUsage: python files/pipeline.py [IMAGE_PATH] [OPTIONAL: COMP_MODEL] [OPTIONAL: INS_MODEL]")
        print(f"Example: python files/pipeline.py test.jpg")
        sys.exit(1)

    img_path   = sys.argv[1]
    comp_path  = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_COMP_MODEL
    ins_path   = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_INS_MODEL
    shed_path  = DEFAULT_SHED_MODEL # Hardcoded for now
    crop_path  = None

    print(f"🚀 Starting Pipeline...")
    print(f"   Image     : {img_path}")
    print(f"   Component : {comp_path}")
    print(f"   Insulator : {ins_path}")
    print(f"   Shed/Disc : {shed_path}")
    
    if not Path(comp_path).exists():
        print(f"❌ Error: Component model not found at {comp_path}")
        sys.exit(1)
    if not Path(ins_path).exists():
        print(f"❌ Error: Insulator model not found at {ins_path}")
        sys.exit(1)
    if not Path(shed_path).exists():
        print(f"❌ Error: Shed/Disc model not found at {shed_path}")
        sys.exit(1)

    pipeline = InfrastructurePipeline(comp_path, ins_path, shed_path, crop_path)
    result   = pipeline.predict(img_path, visualize=True)

    print("\n" + "=" * 55)
    print(f"FINAL CLASS  : {result.final_class}")
    print(f"REASON       : {result.reason}")
    print(f"VOLTAGE      : {result.voltage}")
    print(f"CONFIDENCE   : {result.confidence}")
    print(f"CROSSARM     : {result.crossarm_count}x {result.crossarm_shape}")
    print(f"CONDUCTORS   : {result.conductor_count}")
    print(f"FLAGS        : {result.flags}")
    print(f"SIGNALS USED : {result.signals_used}")
    if result.adjustment_faults:
        print(f"\nADJUSTMENT FAULTS ({len(result.adjustment_faults)}):")
        for f in result.adjustment_faults:
            icon = "🔴" if f["severity"] == "fault" else "🟡"
            print(f"  {icon} [{f['severity'].upper()}] {f['component']}: {f['note']}")
    else:
        print("\nADJUSTMENT FAULTS: none detected")
    if result.insulators:
        print("\nINSULATOR DETAILS:")
        for i, ins in enumerate(result.insulators):
            print(f"  [{i}] type={ins.type_final}"
                  f"  heuristic={ins.type_heuristic}"
                  f"  sheds={ins.shed_count}"
                  f"  voltage={ins.voltage}"
                  f"  AR={ins.aspect_ratio}"
                  f"  conf={ins.type_confidence}")
    if result.pole_orientation:
        po = result.pole_orientation
        print(f"\nPOLE: {po.pole_type}  lean={po.lean_angle_deg}°  {po.note}")
    print("=" * 55)
