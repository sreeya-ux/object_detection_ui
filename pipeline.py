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
    THRESHOLD_INSULATOR, THRESHOLD_CROSSARM, THRESHOLD_POLE, THRESHOLD_CONDUCTOR
)
from insulator_classifier import InsulatorClassifier, InsulatorResult
from crossarm_classifier  import (
    classify_pole_orientation, classify_crossarm_shape,
    aggregate_crossarm_results, PoleOrientationResult, CrossarmResult
)
from rule_engine import classify_pole, ComponentSignals, ClassificationResult


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
        crop_classifier_path: Optional[str] = None,
        conf: float = DETECTION_CONF,
        iou:  float = DETECTION_IOU,
    ):
        """
        Args:
            component_model_path : path to your trained component YOLO (.pt)
            insulator_model_path : path to specialized insulator detection model (.pt)
            shed_model_path      : path to your shed-count model (.pt)
            crop_classifier_path : path to optional insulator crop classifier (.pt)
            conf                 : detection confidence threshold
            iou                  : NMS IoU threshold
        """
        print("Loading component model...")
        self.component_model = YOLO(component_model_path)
        self.is_obb = "obb" in component_model_path.lower()

        print("Loading dedicated insulator detector...")
        self.insulator_detector = YOLO(insulator_model_path)

        print("Loading insulator classifier (shed model + crop classifier)...")
        self.insulator_clf = InsulatorClassifier(
            shed_model_path      = shed_model_path,
            crop_classifier_path = crop_classifier_path,
        )

        self.conf = conf
        self.iou  = iou
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

        # ── Step 1: Run component detector (Multiscale for thin conductors) ────
        # 640 & 1280 cover most structural elements.
        # 1600 is used specifically to resolve thin wires (conductors).
        raw640  = self.component_model(image_path, conf=self.conf, iou=self.iou, verbose=False, imgsz=640)
        raw1280 = self.component_model(image_path, conf=self.conf, iou=self.iou, verbose=False, imgsz=1280)
        raw1600 = self.component_model(image_path, conf=self.conf, iou=self.iou, verbose=False, imgsz=1600)
        
        # ── Step 2: Run specialized insulator detector (Hardware Detail) ──
        # Boost sensitivity and resolution for tiny insulators.
        raw_insulator = self.insulator_detector(image_path, conf=THRESHOLD_INSULATOR, imgsz=1600, verbose=False)

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
        for result in list(raw640) + list(raw1280) + list(raw1600):
            obb   = result.obb   if hasattr(result, "obb")   and result.obb else None
            boxes = result.boxes if hasattr(result, "boxes") and result.boxes else None

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
                         self._categorise(cls_name, box, conf_val, angle_deg, insulator_boxes, pole_boxes_raw, crossarm_boxes, conductor_boxes, street_light_boxes, other_boxes, flags, polygon=poly)
                    elif _match_keyword(cls_name, "crossarm") and conf_val >= THRESHOLD_CROSSARM:
                         self._categorise(cls_name, box, conf_val, angle_deg, insulator_boxes, pole_boxes_raw, crossarm_boxes, conductor_boxes, street_light_boxes, other_boxes, flags, polygon=poly)
                    elif conf_val >= 0.05: # Default for other flags like DTR/AB Cable
                         self._categorise(cls_name, box, conf_val, angle_deg, insulator_boxes, pole_boxes_raw, crossarm_boxes, conductor_boxes, street_light_boxes, other_boxes, flags, polygon=poly)

            if boxes is not None and len(boxes) > 0:
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
                         self._categorise(cls_name, box, conf_val, angle_deg, insulator_boxes, pole_boxes_raw, crossarm_boxes, conductor_boxes, street_light_boxes, other_boxes, flags, polygon=poly)
                    elif _match_keyword(cls_name, "crossarm") and conf_val >= THRESHOLD_CROSSARM:
                         self._categorise(cls_name, box, conf_val, angle_deg, insulator_boxes, pole_boxes_raw, crossarm_boxes, conductor_boxes, street_light_boxes, other_boxes, flags, polygon=poly)
                    elif conf_val >= 0.05:
                         self._categorise(cls_name, box, conf_val, angle_deg, insulator_boxes, pole_boxes_raw, crossarm_boxes, conductor_boxes, street_light_boxes, other_boxes, flags, polygon=poly)
        
        # Process specialized insulator detector output (Final Insulator Detections)
        for result in raw_insulator:
            boxes = result.boxes if hasattr(result, "boxes") and result.boxes else None
            if boxes is not None and len(boxes) > 0:
                for box_obj in boxes:
                    cls_name = self.insulator_detector.names[int(box_obj.cls)]
                    conf_val = float(box_obj.conf)
                    b        = box_obj.xyxy[0].cpu().numpy()
                    box      = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
                    bw, bh   = b[2] - b[0], b[3] - b[1]
                    angle_deg = None

                    # Route to correct stream
                    poly = [[int(b[0]), int(b[1])], [int(b[2]), int(b[1])], [int(b[2]), int(b[3])], [int(b[0]), int(b[3])]]
                    
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
        if not pole_boxes_raw:
            inferred = self._infer_pole_if_missing(insulator_boxes, crossarm_boxes, img_h, img_w)
            if inferred:
                b = inferred[0]
                poly = [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]
                pole_boxes_raw.append((inferred[0], inferred[1], inferred[2], poly))
                flags["inferred_pole"] = True

        # ── Classify each insulator ───────────────────────────
        insulator_results = []
        for box, conf_val, angle_deg, polygon in insulator_boxes:
            ins_result = self.insulator_clf.classify(
                img_original, box, conf_val, obb_angle_deg=angle_deg
            )
            ins_result.obb_polygon = polygon
            insulator_results.append(ins_result)

        all_poles = []
        pole_result = None
        if pole_boxes_raw:
            # Sort by ACTUAL AREA (x2-x1) * (y2-y1) to find the primary pole
            pole_boxes_raw.sort(key=lambda x: (x[0][2]-x[0][0])*(x[0][3]-x[0][1]), reverse=True)
            
            for i, (p_box, p_conf, p_angle, p_poly) in enumerate(pole_boxes_raw):
                pr = classify_pole_orientation(p_box, p_angle)
                pr.detection_conf = p_conf
                pr.obb_polygon = p_poly
                all_poles.append(pr)
                
                # The first (largest) pole is the primary one for rule engine classification
                if i == 0:
                    pole_result = pr

        # ── Classify crossarm shapes ──────────────────────────
        crossarm_results = []
        for box, conf, ang, poly, native_cls in crossarm_boxes:
            cr = classify_crossarm_shape(
                box,
                [c[0] for c in conductor_boxes],
                [p[0] for p in pole_boxes_raw],
                (img_h, img_w),
                obb_angle_deg=ang,
                insulator_results=insulator_results,
                native_class=native_cls
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
        pin_insulators = [i for i in insulator_results if i.type_final == "pin"]
        disc_insulators = [i for i in insulator_results if i.type_final == "disc"]
        
        max_v = "unknown"
        max_s = 0
        max_c = "low"
        
        if pin_insulators:
            shed_counts = [i.shed_count for i in pin_insulators]
            majority_s  = Counter(shed_counts).most_common(1)[0][0]
            from insulator_classifier import ShedCounter
            max_v = ShedCounter.to_voltage(majority_s, "pin")
            max_s = majority_s
            max_c = "high" if any(i.type_confidence == "high" for i in pin_insulators) else "medium"
            
        elif disc_insulators:
            shed_counts = [i.shed_count for i in disc_insulators]
            majority_s  = max(shed_counts) if shed_counts else 0 
            from insulator_classifier import ShedCounter
            max_v = ShedCounter.to_voltage(majority_s, "disc")
            max_s = majority_s
            max_c = "high" if any(i.type_confidence == "high" for i in disc_insulators) else "medium"

        signals = ComponentSignals(
            insulator_type    = "pin" if any(i.type_final == "pin" for i in insulator_results) else "unknown",
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
            # Include all scales in visualization
            raw_combined = list(raw640) + list(raw1280) + list(raw1600) + list(raw_insulator)
            vis = self._draw(img, pipeline_result, raw_combined)
            out = save_path or (Path(image_path).stem + "_result.jpg")
            cv2.imwrite(str(out), vis)
            print(f"Saved: {out}")

        return pipeline_result

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
        polygon=None
    ):
        """Routes a detection into the right typed list, including polygon data."""
        det = (box, conf_val, angle_deg, polygon)
        if _match_keyword(cls_name, "insulator"):
            insulator_boxes.append(det)
        elif _match_keyword(cls_name, "pole"):
            pole_boxes_raw.append(det)
        elif _match_keyword(cls_name, "crossarm"):
            native = cls_name.lower()
            # Structural Guard: Crossarms MUST be horizontal. 
            # We allow a small tolerance (1.2x) for significantly tilted arms.
            w, h = box[2] - box[0], box[3] - box[1]
            if (w * 1.2) > h or "t_rising" in native or "side_arm" in native:
                crossarm_boxes.append((box, conf_val, angle_deg, polygon, native))
            else:
                # If it's vertical but high confidence, maybe it's a pole?
                if conf_val > 0.40:
                    pole_boxes_raw.append(det)
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
        elif _match_keyword(cls_name, "lattice_frame"):
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
            
        # Sort by confidence descending
        sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
        keep = []
        
        while sorted_items:
            best = sorted_items.pop(0)
            keep.append(best)
            
            # Filter remaining items
            remaining = []
            for item in sorted_items:
                overlap = self._calculate_max_overlap(best[0], item[0])
                if overlap < iou_threshold:
                    remaining.append(item)
            sorted_items = remaining
            
        return keep

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

        # Draw banner
        if result.pole_orientation:
            po = result.pole_orientation
            x1, y1, x2, y2 = po.box
            is_inferred = result.flags.get("inferred_pole", False)
            color = (255, 128, 0) if is_inferred else (220, 180, 50)
            
            if is_inferred:
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
                label = f"[INFERRED] {po.pole_type}"
            else:
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                label = f"{po.pole_type} ({po.detection_conf:.2f})"
                
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
                    f"thickness={result.flags.get('avg_thickness_px', 0)}px",
                    (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (120, 120, 120), 1)

        return np.vstack([banner, vis])


# ── CLI ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    
    # ── Configuration: Default Model Paths ────────────────────
    # These will be used if no command line arguments are provided.
    # Paths are relative to the project root (D:\NEW_ASAKTA\dry).
    ROOT = Path(__file__).parent.parent.absolute()
    DEFAULT_COMP_MODEL = str(ROOT / "best_whole.pt")
    DEFAULT_INS_MODEL  = str(ROOT / "best_insulator.pt")
    DEFAULT_SHED_MODEL = str(ROOT / "best_disc.pt")

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
