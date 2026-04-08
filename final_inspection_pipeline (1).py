# d:\NEW_ASAKTA\dry\final_inspection_pipeline.py
import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import sys
import os
from PIL import Image, ImageOps

# --- CONFIGURE PATHS ---
CURR_DIR = Path(__file__).parent.absolute()
sys.path.append(str(CURR_DIR / "files"))
from pipeline import InfrastructurePipeline, PipelineResult
from crossarm_classifier import classify_pole_orientation, classify_crossarm_shape
import segmentation_models_pytorch as smp

# --- THE UNIFIED MASTER SET ---
MODELS_DIR = CURR_DIR / "new_models"
BACKBONE_OBB    = MODELS_DIR / "obb_train" / "weights" / "best.pt" 
INSULATOR_PATH  = MODELS_DIR / "insulator_finetuned" / "content" / "runs" / "detect" / "insulator_finetuned" / "weights" / "best.pt"
UNET_MASK_PATH  = MODELS_DIR / "best_cable_unet.pth"
DISC_COUNT_PATH = MODELS_DIR / "best_disc.pt"

class MasterInspectionPipeline(InfrastructurePipeline):
    def __init__(self):
        print("🛠️ Initializing Unified Master Infrastructure Tool...")
        super().__init__(str(BACKBONE_OBB), str(INSULATOR_PATH), str(DISC_COUNT_PATH))
        
        print("🧶 Loading Conductor Instance Seg (U-NET)...")
        self.cable_model = smp.Unet("resnet34", in_channels=3, classes=1)
        self.cable_model.load_state_dict(torch.load(UNET_MASK_PATH, map_location='cpu'))
        self.cable_model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cable_model.to(self.device)

    def _merge_boxes(self, boxes_with_conf):
        if not boxes_with_conf: return []
        # Simplified: If many arm boxes overlap, merge them into one giant box
        # Group by proximity (approximate NMS but merging coordinates)
        merged = []
        # Sort by confidence
        sorted_boxes = sorted(boxes_with_conf, key=lambda x: x[1], reverse=True)
        
        while sorted_boxes:
            base_box, base_conf, base_ang = sorted_boxes.pop(0)
            cluster = [base_box]
            remaining = []
            
            for b, c, a in sorted_boxes:
                # Calculate simple IoU or proximity
                # We'll merge ANY arm box that overlaps at all
                if self._get_iou(base_box, b) > 0.05:
                    cluster.append(b)
                else:
                    remaining.append((b, c, a))
            
            # Create a bounding box covering the whole cluster
            cluster_np = np.array(cluster)
            xmin = np.min(cluster_np[:, 0])
            ymin = np.min(cluster_np[:, 1])
            xmax = np.max(cluster_np[:, 2])
            ymax = np.max(cluster_np[:, 3])
            
            merged.append(([xmin, ymin, xmax, ymax], base_conf, base_ang))
            sorted_boxes = remaining
        return merged

    def _get_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        return interArea / float(boxAArea + boxBArea - interArea)

    def predict(self, image_path, visualize=True):
        pil_img = Image.open(image_path)
        pil_img = ImageOps.exif_transpose(pil_img)
        img_original = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        img_h, img_w = img_original.shape[:2]
        temp_path = str(CURR_DIR / "temp_oriented.jpg")
        cv2.imwrite(temp_path, img_original)
        
        # 1. STRUCTURAL DETECTION
        print("🏗️ 1. Identifying Structural OBB Elements...")
        raw_struct = self.component_model(temp_path, imgsz=1280, conf=0.01, verbose=False)
        
        raw_pole_list = []
        raw_arm_list = []
        for r in raw_struct:
            if hasattr(r, 'obb') and r.obb:
                for i in range(len(r.obb)):
                    nm   = self.component_model.names[int(r.obb.cls[i])].lower()
                    conf = float(r.obb.conf[i])
                    box  = r.obb.xyxy[i].cpu().numpy().astype(int)
                    ang  = np.degrees(float(r.obb.xywhr[i][4]))
                    if "pole" in nm: raw_pole_list.append((box, conf, ang))
                    elif "crossarm" in nm: raw_arm_list.append((box, conf, ang))

        # MERGE OVERLAPPING BOXES INTO ONE CLEAN RESULT
        pole_boxes = self._merge_boxes(raw_pole_list)
        crossarm_boxes = self._merge_boxes(raw_arm_list)

        # 2. SEQUENTIAL SCANNING 
        raw_ins = self.insulator_detector(temp_path, imgsz=1280, conf=0.15, verbose=False)
        ins_res = [self.insulator_clf.classify(img_original, b.xyxy[0].cpu().numpy().astype(int), float(b.conf)) for r in raw_ins for b in (r.boxes or [])]
        
        inp = cv2.resize(img_original, (512, 512)).transpose(2, 0, 1) / 255.0
        inp_t = torch.tensor(inp[None, ...], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            mask_out = torch.sigmoid(self.cable_model(inp_t)).cpu().numpy()[0, 0]
        mask_full = (cv2.resize(mask_out, (img_w, img_h)) > 0.5).astype(np.uint8) * 255
        
        # 3. ASSEMBLE RULE ENGINE 
        p_res = None
        if pole_boxes:
            pole_boxes.sort(key=lambda x: (x[0][2]-x[0][0])*(x[0][3]-x[0][1]), reverse=True)
            p_res = classify_pole_orientation(pole_boxes[0][0], pole_boxes[0][2])
            p_res.detection_conf = pole_boxes[0][1]

        t_crossarms = [classify_crossarm_shape(b, [], [], (img_h, img_w), a, ins_res) for b,c,a in crossarm_boxes]
        for idx, (b,c,a) in enumerate(crossarm_boxes): t_crossarms[idx].detection_conf = c
        
        from rule_engine import classify_pole, ComponentSignals
        signals = ComponentSignals(
            insulator_type="pin" if ins_res else "none",
            insulator_voltage=ins_res[0].voltage if ins_res else "none",
            shed_count=ins_res[0].shed_count if ins_res else 0,
            pole_type=p_res.pole_type if p_res else "vertical_pole",
            crossarm_shape=t_crossarms[0].shape if t_crossarms else "none"
        )
        final = classify_pole(signals)
        res_obj = PipelineResult(
            final_class=final.final_class, class_id=final.class_id, reason=final.reason,
            voltage=final.voltage, confidence=final.confidence, signals_used=[],
            insulators=ins_res, pole_orientation=p_res, crossarms=t_crossarms,
            crossarm_shape=signals.crossarm_shape, conductor_count="InstMask"
        )
        
        if visualize:
            vis = self._draw(img_original, res_obj, [])
            overlay = vis[75:].copy(); overlay[mask_full > 0] = (255, 255, 0)
            vis[75:] = cv2.addWeighted(vis[75:], 0.7, overlay, 0.3, 0)
            cv2.imwrite(f"{Path(image_path).stem}_final_result.jpg", vis)
            print(f"✅ Unified Report Saved (One Crossarm Box).")
        
        if os.path.exists(temp_path): os.remove(temp_path)
        return res_obj

if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(1)
    MasterInspectionPipeline().predict(sys.argv[1])
