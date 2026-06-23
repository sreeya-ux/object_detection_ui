import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from pathlib import Path
import math
import time
from collections import defaultdict, Counter

# Internal imports
from rule_engine import classify_pole, ComponentSignals, PoleResult, InsulatorResult, PipelineResult
from insulator_classifier import InsulatorClassifier
from runtime_device import inference_device

from config import (
    POLE_CLASSES,
    DETECTION_CONF,
    DETECTION_IOU,
    THRESHOLD_POLE,
    THRESHOLD_INSULATOR,
    THRESHOLD_CROSSARM,
    THRESHOLD_CONDUCTOR,
    GLOBAL_TILT_MAX_DEG,
    HT_LT_HEIGHT_THRESHOLD
)

def _match_keyword(name: str, keyword: str) -> bool:
    return keyword.lower() in name.lower()

def classify_pole_orientation(box, angle_deg, is_strut=False, tilt_compensation=0.0) -> PoleResult:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    
    # Calculate lean relative to vertical (90 deg)
    actual_angle = angle_deg if angle_deg is not None else (90.0 if h > w else 0.0)
    lean_angle = abs(actual_angle - 90.0) - abs(tilt_compensation)
    lean_angle = max(0.0, lean_angle)
    
    p_type = "strut_pole" if is_strut else "vertical_pole"
    
    return PoleResult(
        pole_type=p_type,
        lean_angle_deg=round(lean_angle, 1),
        box=box,
        note=f"Lean: {lean_angle:.1f} deg"
    )

def _safe_yolo_load(path: str) -> "YOLO":
    """Load a YOLO model and silently skip fuse() if it fails.
    
    Ultralytics 8.3+ changed the internal Conv.fuse() behaviour; older .pt
    weights (pre-8.x) lack the 'bn' attribute that the new fuse() expects.
    We monkey-patch the model's fuse method so it degrades gracefully instead
    of raising AttributeError.
    """
    model = YOLO(path)
    _orig_fuse = model.model.fuse if hasattr(model.model, 'fuse') else None
    
    def _safe_fuse(*args, **kwargs):
        try:
            if _orig_fuse:
                return _orig_fuse(*args, **kwargs)
        except AttributeError:
            print(f"[WARNING] fuse() skipped for {path} (old weights, no 'bn' attr) — running unfused.")
    
    if hasattr(model.model, 'fuse'):
        model.model.fuse = _safe_fuse
    
    # Also patch all Conv sub-modules individually so inference doesn't crash
    try:
        import torch.nn as nn
        for m in model.model.modules():
            if type(m).__name__ == 'Conv' and not hasattr(m, 'bn'):
                m.bn = nn.Identity()
    except Exception as e:
        print(f"[WARNING] Conv patch skipped: {e}")
    
    return model


class InfrastructurePipeline:
    def __init__(self, comp_model, hardware_model, shed_model, crop_clf=None, insulator_model=None, hardware_model_extra=None):
        # comp_model: only for poles (main_pole, strut_pole)
        self.component_model = _safe_yolo_load(comp_model) if comp_model else None
        # hardware_model: component hardware such as arms, cleats, switches, lights, DTR.
        self.hardware_model = _safe_yolo_load(hardware_model)
        # hardware_model_extra: optional secondary component hardware model for ensembling
        self.hardware_model_extra = _safe_yolo_load(hardware_model_extra) if hardware_model_extra else None
        # insulator_model: dedicated insulator detector/classifier model.
        self.insulator_model = _safe_yolo_load(insulator_model) if insulator_model else None
        self.insulator_clf = InsulatorClassifier(shed_model, crop_clf)
        self.conf, self.iou = DETECTION_CONF, DETECTION_IOU


    def predict(self, image_path, visualize=True, save_path=None, fast_mode=False) -> PipelineResult:
        img_original = cv2.imread(image_path)
        if img_original is None: raise FileNotFoundError(image_path)
        img = self._enhance_image(img_original)
        img_h, img_w = img.shape[:2]
        import gc, torch
        active_device = inference_device()

        # 1. Structural Detect (Poles)
        raw_structural = []
        structural_imgsz = 512 if fast_mode else 800
        if self.component_model:
            raw_structural = list(self.component_model(image_path, conf=self.conf, iou=self.iou, verbose=False, imgsz=structural_imgsz, device=active_device))
        
        # 2. Hardware Detect (arms, cleats, switches, lights, DTR)
        raw_hardware = []
        try:
            tile_size = 640 if fast_mode else 1024
            overlap = 0.50
            hw_res = self.hardware_model(image_path, conf=0.05, imgsz=tile_size, verbose=False, device=active_device)
            raw_hardware.extend(list(hw_res))
            
            if self.hardware_model_extra:
                hw_extra_res = self.hardware_model_extra(image_path, conf=0.05, imgsz=tile_size, verbose=False, device=active_device)
                raw_hardware.extend(list(hw_extra_res))
            
            if not fast_mode and (img_w > 1500 or img_h > 1500):
                step = int(tile_size * (1 - overlap))
                for y in range(0, img_h, step):
                    for x in range(0, img_w, step):
                        x2, y2 = min(x + tile_size, img_w), min(y + tile_size, img_h)
                        x1, y1 = max(0, x2 - tile_size), max(0, y2 - tile_size)
                        tile_crop = img[y1:y2, x1:x2]
                        tile_res = self.hardware_model(tile_crop, conf=0.05, imgsz=tile_size, verbose=False, device=active_device)
                        for r in tile_res:
                            if r.boxes:
                                for b_obj in r.boxes:
                                    cls_id = int(b_obj.cls)
                                    conf = float(b_obj.conf)
                                    b = b_obj.xyxy[0].cpu().numpy().copy()
                                    b[0] += x1; b[2] += x1
                                    b[1] += y1; b[3] += y1
                                    raw_hardware.append((tuple(b.astype(int)), conf, cls_id))
                        
                        if self.hardware_model_extra:
                            tile_extra_res = self.hardware_model_extra(tile_crop, conf=0.05, imgsz=tile_size, verbose=False, device=active_device)
                            for r in tile_extra_res:
                                if r.boxes:
                                    for b_obj in r.boxes:
                                        cls_id = int(b_obj.cls)
                                        conf = float(b_obj.conf)
                                        b = b_obj.xyxy[0].cpu().numpy().copy()
                                        b[0] += x1; b[2] += x1
                                        b[1] += y1; b[3] += y1
                                        raw_hardware.append((tuple(b.astype(int)), conf, cls_id))
                                        
                        del tile_crop; gc.collect()
                        if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception as e:
            print(f"Hardware detection error: {e}")

        # 2b. Dedicated insulator detection
        raw_insulators = []
        if self.insulator_model:
            try:
                ins_res = self.insulator_model(image_path, conf=0.05, imgsz=(640 if fast_mode else 1024), verbose=False, device=active_device)
                raw_insulators.extend(list(ins_res))
            except Exception as e:
                print(f"Insulator detection error: {e}")

        # 3. Parse
        ins_b, pole_b, arm_b, cond_b, other_b, flags = [], [], [], [], [], defaultdict(bool)
        
        # Process Structural Model (Poles & Conductors)
        for res in raw_structural:
            if res.boxes:
                for i, b_obj in enumerate(res.boxes):
                    cls = self.component_model.names[int(b_obj.cls)]
                    conf = float(b_obj.conf)
                    b = b_obj.xyxy[0].cpu().numpy().astype(int)
                    
                    poly = None
                    if hasattr(res, 'masks') and res.masks is not None:
                        poly = res.masks.xy[i].tolist()
                        
                    if _match_keyword(cls, "pole") and conf > THRESHOLD_POLE: 
                        pole_b.append((tuple(b), conf, None, poly, "strut" in cls.lower()))
                    elif _match_keyword(cls, "conductor") and conf > THRESHOLD_CONDUCTOR:
                        cond_b.append((tuple(b), conf))

        # Process Hardware Model (component classes + Polygons)
        for res in raw_hardware:
            if hasattr(res, 'boxes') and res.boxes:
                for i, b_obj in enumerate(res.boxes):
                    cls_id = int(b_obj.cls)
                    conf = float(b_obj.conf)
                    b = b_obj.xyxy[0].cpu().numpy().astype(int)
                    
                    # Extract polygon if it exists (segmentation model)
                    poly = None
                    if hasattr(res, 'masks') and res.masks is not None:
                        poly = res.masks.xy[i].tolist()
                        
                    self._categorise_12class(cls_id, tuple(b), conf, ins_b, arm_b, other_b, poly, pole_b, cond_b)
            elif isinstance(res, tuple):
                b, conf, cls_id = res
                self._categorise_12class(cls_id, b, conf, ins_b, arm_b, other_b, None, pole_b, cond_b)

        # Process dedicated insulator model.
        for res in raw_insulators:
            if hasattr(res, 'boxes') and res.boxes:
                for i, b_obj in enumerate(res.boxes):
                    conf = float(b_obj.conf)
                    if conf <= THRESHOLD_INSULATOR:
                        continue
                    b = b_obj.xyxy[0].cpu().numpy().astype(int)
                    poly = None
                    if hasattr(res, 'masks') and res.masks is not None:
                        poly = res.masks.xy[i].tolist()
                    cls_id = int(b_obj.cls)
                    detector_class = self._normalise_model_name(self.insulator_model.names[cls_id]).upper()
                    ins_b.append((tuple(b), conf, detector_class, poly, False))

        # 4. Finalize
        ins_b = self._nms(ins_b, 0.35) # Stricter NMS to prevent insulator overlaps
        pole_b = self._nms(pole_b, 0.45)
        arm_b = self._nms(arm_b, 0.40) # 0.40 NMS prevents double-boxes without deleting adjacent arms
        cond_b = self._nms(cond_b, 0.30) # Suppress overlapping conductor boxes
        other_b = self._nms(other_b, 0.40) # Suppress duplicate hardware/special clamp/street light detections
        
        ins_results = []
        for b, c, detector_class, poly, _ in ins_b:
            if fast_mode:
                cls_text = (detector_class or "").lower()
                type_final = "disc" if "disc" in cls_text else "pin"
                voltage = "33kV" if type_final == "disc" else "11kV"
                ir = InsulatorResult(
                    box=tuple(b),
                    confidence=float(c),
                    type_final=type_final,
                    voltage=voltage,
                    shed_count=0,
                    classification_conf=float(c)
                )
            else:
                ir = self.insulator_clf.classify(img_original, b, c)
            ir.obb_polygon = poly
            ir.detection_conf = c # UI expects this
            ir.detector_class = detector_class
            ins_results.append(ir)
            
        all_poles_res = []
        for b, c, _, poly, is_strut in pole_b:
            # Calculate actual lean angle from mask if available
            lean_angle = 0.0
            if poly:
                pts = np.array(poly)
                if len(pts) > 10:
                    y_coords = pts[:, 1]
                    x_coords = pts[:, 0]
                    try:
                        slope, intercept = np.polyfit(y_coords, x_coords, 1)
                        # Angle in degrees = |atan(slope)| converted to degrees
                        lean_angle = abs(math.degrees(math.atan(slope)))
                    except Exception as e:
                        print(f"[WARNING] Pole lean angle fitting failed: {e}")
                        
            pr = classify_pole_orientation(b, 90.0 - lean_angle, is_strut=is_strut)
            pr.obb_polygon = poly
            pr.detection_conf = c
            all_poles_res.append(pr)

        all_arms_res = []
        # Find main pole for context checks
        main_pole = None
        if all_poles_res:
            main_pole = max(all_poles_res, key=lambda p: (p.box[2]-p.box[0])*(p.box[3]-p.box[1]))

        for b, c, name, poly, _ in arm_b:
            # Map standard model names to frontend/UI classes
            if name == "side_arm":
                name = "side_arm_channel"
            elif name == "tapping_arm":
                name = "tapping_channel"
            elif name == "v_cross_arm":
                name = "v_cross"
            elif name == "t_rising":
                name = "t_rising"
            # ── Trust the YOLO model's raw visual predictions directly ──────────


            ar_obj = PoleResult(pole_type=name, lean_angle_deg=0.0, box=b, obb_polygon=poly)
            ar_obj.detection_conf = c
            ar_obj.shape = "straight"
            all_arms_res.append(ar_obj)

        # Resolve the model's common street-light/special-clamp confusion using
        # pole context before splitting dedicated UI classes.
        st_lights = []
        raw_final_others = []
        for box, conf, name, poly in other_b:
            name_lower = name.lower()
            if name_lower == "street_light" and self._is_compact_pole_mounted_hardware(box, all_poles_res):
                print(
                    "[IMAGE-HARDWARE-RESOLVE] "
                    f"raw=street_light resolved=special_clamp conf={float(conf):.3f} "
                    f"bbox={[int(v) for v in box]} reason=compact_pole_overlap",
                    flush=True
                )
                name = "special_clamp"
                name_lower = name
            if name_lower == "street_light" or name_lower.startswith("lamp_") or name_lower in {"lamp", "lamp_head"}:
                st_lights.append((box, conf, poly))
            else:
                raw_final_others.append((name, box, conf, poly))

        # Filter to keep at most one special_clamp (the one with the highest confidence)
        final_others = []
        best_clamp = None
        for item in raw_final_others:
            name, box, conf, poly = item
            if name == "special_clamp":
                if best_clamp is None or conf > best_clamp[2]:
                    best_clamp = item
            else:
                final_others.append(item)
        if best_clamp is not None:
            final_others.append(best_clamp)

        pole_id_str = "Not Found"

        arm_shape = "none"
        if all_arms_res:
            arm_types = [a.pole_type for a in all_arms_res]
            if "v_cross" in arm_types:
                arm_shape = "v_arm"
            elif "t_rising" in arm_types:
                arm_shape = "t_raising"
            elif "tapping_channel" in arm_types:
                arm_shape = "straight"
            elif "side_arm_channel" in arm_types:
                arm_shape = "side_arm"

        signals = ComponentSignals(
            insulator_type = ins_results[0].type_final if ins_results else "unknown",
            insulator_voltage = ins_results[0].voltage if ins_results else "unknown",
            pole_type = main_pole.pole_type if main_pole else "vertical_pole",
            lean_angle_deg = main_pole.lean_angle_deg if main_pole else 0.0,
            conductor_count = len(cond_b),
            crossarm_count = len(all_arms_res),
            crossarm_shape = arm_shape
        )
        final = classify_pole(signals)
        
        res_obj = PipelineResult(
            final_class=final.final_class, class_id=final.class_id, reason=final.reason,
            voltage=final.voltage, confidence=final.confidence, signals_used=final.signals_used,
            insulators=ins_results,
            all_poles=all_poles_res,
            all_arms=all_arms_res,
            street_lights=st_lights,
            pole_orientation=main_pole,
            conductor_count=len(cond_b),
            crossarm_count=len(all_arms_res),
            pole_id=pole_id_str,
            flags=dict(flags), adjustment_faults=final.faults
        )
        res_obj.others = final_others
        
        if visualize:
            vis = self._draw(img_original, res_obj)
            cv2.imwrite(save_path or "result.jpg", vis)
        return res_obj

    def _normalise_model_name(self, name: str) -> str:
        return name.strip().lower().replace(" ", "_")

    def _categorise_12class(self, cls_id, box, conf, ins_b, arm_b, other_b, poly, pole_b=None, cond_b=None):
        raw_name = self._normalise_model_name(self.hardware_model.names[cls_id])
        name = raw_name
        # Keep raw model labels visible in logs so class mapping mistakes are easy to trace.
        threshold = None
        bucket = "drop"
        kept = False
        if "pole" in name and pole_b is not None:
            threshold = THRESHOLD_POLE
            if conf > THRESHOLD_POLE:
                pole_b.append((box, conf, None, poly, "strut" in name))
                bucket = "pole_b"
                kept = True
        elif ("conductor" in name or "wire" in name or "cable" in name) and cond_b is not None:
            threshold = THRESHOLD_CONDUCTOR
            if conf > THRESHOLD_CONDUCTOR:
                cond_b.append((box, conf))
                bucket = "cond_b"
                kept = True
        elif "insulator" in name or name in {"ins_pin", "ins_disc"}:
            threshold = THRESHOLD_INSULATOR
            if conf > THRESHOLD_INSULATOR:
                ins_b.append((box, conf, None, poly, False))
                bucket = "ins_b"
                kept = True
        elif name in {"v_cross_arm", "tapping_arm", "top_cleat", "side_arm", "t_rising", "box_arm"}:
            # Lower threshold specifically for side arms and tapping arms to ensure detection
            thresh = THRESHOLD_CROSSARM
            if name in ["side_arm", "side_arm_channel", "tapping_arm", "tapping_channel"]:
                thresh = 0.35
            threshold = thresh
            if conf > thresh:
                arm_b.append((box, conf, name, poly, False))
                bucket = "arm_b"
                kept = True
        else:
            # Use lower threshold specifically for stay_set
            thresh = 0.15
            if name == "stay_set":
                thresh = 0.05
            threshold = thresh
            if conf > thresh:
                other_b.append((box, conf, name, poly))
                bucket = "other_b"
                kept = True
        if raw_name in {"street_light", "special_clamp"} or name in {"street_light", "special_clamp"}:
            print(
                "[IMAGE-HARDWARE] "
                f"cls_id={cls_id} raw={raw_name} mapped={name} conf={float(conf):.3f} "
                f"threshold={float(threshold or 0):.2f} kept={kept} bucket={bucket} "
                f"bbox={[int(v) for v in box]}",
                flush=True
            )

    def _nms(self, items, thresh):
        if not items: return []
        items.sort(key=lambda x: x[1], reverse=True)
        keep = []
        while items:
            best = items.pop(0); keep.append(best)
            # Agnostic NMS: suppress ANY box that overlaps, even if it's a different class (e.g., top_cleat vs v_cross_arm)
            items = [i for i in items if self._iou(best[0], i[0]) < thresh]
        return keep

    def _iou(self, b1, b2):
        xA, yA, xB, yB = max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        area1, area2 = (b1[2]-b1[0])*(b1[3]-b1[1]), (b2[2]-b2[0])*(b2[3]-b2[1])
        
        # If one box is mostly inside the other (e.g. duplicate detection of a part of the V-arm), suppress it
        min_area = min(area1, area2)
        if min_area > 0 and (inter / float(min_area)) > 0.60:
            return 1.0  # Force suppression
            
        return inter / float(area1 + area2 - inter + 1e-6)

    def _is_compact_pole_mounted_hardware(self, box, poles):
        if not poles:
            return False
        x1, y1, x2, y2 = [float(v) for v in box]
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        if max(width, height) / min(width, height) > 1.5:
            return False

        box_area = width * height
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        for pole in poles:
            px1, py1, px2, py2 = [float(v) for v in pole.box]
            inter_w = max(0.0, min(x2, px2) - max(x1, px1))
            inter_h = max(0.0, min(y2, py2) - max(y1, py1))
            overlap = (inter_w * inter_h) / box_area
            center_on_pole = px1 <= cx <= px2 and py1 <= cy <= py2
            if center_on_pole or overlap >= 0.20:
                return True
        return False

    def _enhance_image(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
        return cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2BGR)

    def _draw(self, img, res):
        vis = img.copy()
        for i in res.insulators: cv2.rectangle(vis, (i.box[0], i.box[1]), (i.box[2], i.box[3]), (0,255,0), 2)
        if res.pole_orientation:
            p = res.pole_orientation
            cv2.rectangle(vis, (p.box[0], p.box[1]), (p.box[2], p.box[3]), (255,255,255), 2)
        return vis
