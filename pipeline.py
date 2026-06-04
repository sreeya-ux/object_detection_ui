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
    def __init__(self, comp_model, hardware_model, shed_model, crop_clf=None, insulator_model=None):
        # comp_model: only for poles (main_pole, strut_pole)
        self.component_model = _safe_yolo_load(comp_model) if comp_model else None
        # hardware_model: component hardware such as arms, cleats, switches, lights, DTR.
        self.hardware_model = _safe_yolo_load(hardware_model)
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

        # 1. Structural Detect (Poles)
        raw_structural = []
        structural_imgsz = 512 if fast_mode else 800
        if self.component_model:
            raw_structural = list(self.component_model(image_path, conf=self.conf, iou=self.iou, verbose=False, imgsz=structural_imgsz))
        
        # 2. Hardware Detect (arms, cleats, switches, lights, DTR)
        raw_hardware = []
        try:
            tile_size = 640 if fast_mode else 1024
            overlap = 0.50
            hw_res = self.hardware_model(image_path, conf=0.05, imgsz=tile_size, verbose=False)
            raw_hardware.extend(list(hw_res))
            
            if not fast_mode and (img_w > 1500 or img_h > 1500):
                step = int(tile_size * (1 - overlap))
                for y in range(0, img_h, step):
                    for x in range(0, img_w, step):
                        x2, y2 = min(x + tile_size, img_w), min(y + tile_size, img_h)
                        x1, y1 = max(0, x2 - tile_size), max(0, y2 - tile_size)
                        tile_crop = img[y1:y2, x1:x2]
                        tile_res = self.hardware_model(tile_crop, conf=0.05, imgsz=tile_size, verbose=False)
                        for r in tile_res:
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
                ins_res = self.insulator_model(image_path, conf=0.05, imgsz=(640 if fast_mode else 1024), verbose=False)
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

            # ── Geometry-based override (applies to all arm types) ──────────────
            # Only use pole-relative symmetry to disambiguate; do NOT override
            # v_cross based on aspect ratio alone (V arms are naturally wide, ar > 1.0).
            bw = b[2] - b[0]
            bh = b[3] - b[1]
            ar = bw / float(bh) if bh > 0 else 0

            # If we have a polygon for the crossarm, use oriented bounding box to calculate actual tilt-compensated aspect ratio
            if poly:
                try:
                    pts = np.array(poly, dtype=np.float32)
                    if len(pts) >= 3:
                        rect = cv2.minAreaRect(pts)
                        (cx_rect, cy_rect), (w_rect, h_rect), angle_rect = rect
                        actual_w = max(w_rect, h_rect)
                        actual_h = min(w_rect, h_rect)
                        if actual_h > 0:
                            ar = actual_w / float(actual_h)
                except Exception as e:
                    print(f"[WARNING] minAreaRect aspect ratio failed: {e}")

            if all_poles_res:
                cx = (b[0] + b[2]) / 2
                cy = (b[1] + b[3]) / 2

                # Find the best matching pole for this crossarm (the one it is mounted on/intersects)
                active_pole = None
                pole_cx = None
                
                candidates = []
                for p in all_poles_res:
                    px1, py1, px2, py2 = p.box
                    p_cx = (px1 + px2) / 2
                    if hasattr(p, 'obb_polygon') and p.obb_polygon:
                        pts = np.array(p.obb_polygon)
                        if len(pts) > 10:
                            try:
                                slope, intercept = np.polyfit(pts[:, 1], pts[:, 0], 1)
                                p_cx = slope * cy + intercept
                                p_cx = max(float(px1), min(float(px2), float(p_cx)))
                            except Exception:
                                pass
                    
                    # Check horizontal intersection with margin (10% of arm width)
                    margin = (b[2] - b[0]) * 0.10
                    if (b[0] - margin) <= p_cx <= (b[2] + margin):
                        candidates.append((p, p_cx))
                        
                if candidates:
                    # Prioritize vertical poles (non-strut)
                    non_struts = [c for c in candidates if c[0].pole_type == "vertical_pole"]
                    if non_struts:
                        best = max(non_struts, key=lambda c: (c[0].box[2]-c[0].box[0])*(c[0].box[3]-c[0].box[1]))
                        active_pole, pole_cx = best[0], best[1]
                    else:
                        best = max(candidates, key=lambda c: (c[0].box[2]-c[0].box[0])*(c[0].box[3]-c[0].box[1]))
                        active_pole, pole_cx = best[0], best[1]
                else:
                    # Fallback to the main vertical pole
                    non_struts = [p for p in all_poles_res if p.pole_type == "vertical_pole"]
                    if non_struts:
                        active_pole = max(non_struts, key=lambda p: (p.box[2]-p.box[0])*(p.box[3]-p.box[1]))
                    else:
                        active_pole = max(all_poles_res, key=lambda p: (p.box[2]-p.box[0])*(p.box[3]-p.box[1]))
                    
                    px1, py1, px2, py2 = active_pole.box
                    pole_cx = (px1 + px2) / 2
                    if hasattr(active_pole, 'obb_polygon') and active_pole.obb_polygon:
                        pts = np.array(active_pole.obb_polygon)
                        if len(pts) > 10:
                            try:
                                slope, intercept = np.polyfit(pts[:, 1], pts[:, 0], 1)
                                pole_cx = slope * cy + intercept
                                pole_cx = max(float(px1), min(float(px2), float(pole_cx)))
                            except Exception:
                                pass
                                
                px1, py1, px2, py2 = active_pole.box

                # Symmetry: how evenly does the arm span the pole centre?
                left_part  = max(0.0, pole_cx - b[0])
                right_part = max(0.0, b[2]    - pole_cx)
                max_part   = max(left_part, right_part)
                sym_ratio  = min(left_part, right_part) / max_part if max_part > 0 else 0.0

                if name == "v_cross":
                    # Trust the model's v_cross prediction unless the box is
                    # clearly one-sided (strongly asymmetric → it's actually a side arm)
                    if sym_ratio < 0.40:
                        name = "side_arm_channel"
                    # Or if the box is extremely flat/thin (high aspect ratio → straight tapping channel)
                    elif ar >= 5.0:
                        name = "tapping_channel"
                    # else: keep v_cross — a real V arm IS wide (ar > 1 is expected)

                elif name == "tapping_channel":
                    # Tapping channel predicted by model, refine with geometry
                    if sym_ratio < 0.40:
                        name = "side_arm_channel"
                    elif cy < py1 + (py2 - py1) * 0.05:
                        name = "t_rising"

                elif name in ("side_arm_channel", "side_arm"):
                    # If the model calls it side_arm but symmetry says it's centred,
                    # upgrade it to tapping_channel
                    if sym_ratio >= 0.55:
                        name = "tapping_channel"


            ar_obj = PoleResult(pole_type=name, lean_angle_deg=0.0, box=b, obb_polygon=poly)
            ar_obj.detection_conf = c
            ar_obj.shape = "straight"
            all_arms_res.append(ar_obj)

        # Extract street lights from 'other_b' specifically for the UI if needed
        st_lights = []
        final_others = []
        for name, box, conf, poly in other_b:
            if "street_light" in name.lower() or "lamp" in name.lower():
                st_lights.append((box, conf, poly))
            else:
                final_others.append((name, box, conf, poly))

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
        # The channel_12class_v2 checkpoint has these two semantic labels reversed
        # in practice: clamp hardware is emitted as street_light and vice versa.
        if name == "street_light":
            name = "special_clamp"
        elif name == "special_clamp":
            name = "street_light"
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
                other_b.append((name, box, conf, poly))
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
