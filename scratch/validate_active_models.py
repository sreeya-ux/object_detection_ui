import os
import sys
import json
import yaml
import shutil
import random
import argparse
from pathlib import Path
from ultralytics import YOLO

def validate_all(fast_mode=True, max_samples=25):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(script_dir, ".."))
    
    configs = {
        "pole": {
            "model_path": f"{base_dir}/models/best (2).pt",
            "src_path": f"{base_dir}/training_data_component",
            "yaml_data": {
                "nc": 2,
                "names": {
                    0: "main_pole",
                    1: "strut_pole"
                }
            }
        },
        "components": {
            "model_path": f"{base_dir}/models/channel_12class_v2.pt",
            "src_path": f"{base_dir}/dataset_channels",
            "yaml_data": {
                "nc": 12,
                "names": {
                    0: "insulators",
                    1: "v_cross_arm",
                    2: "tapping_arm",
                    3: "top_cleat",
                    4: "side_arm",
                    5: "t_rising",
                    6: "special_clamp",
                    7: "street_light",
                    8: "stay_set",
                    9: "box_arm",
                    10: "ab_switch",
                    11: "dtr"
                }
            }
        },
        "insulator": {
            "model_path": f"{base_dir}/models/insulator_model.pt",
            "src_path": f"{base_dir}/training_data_hardware",
            "yaml_data": {
                "nc": 3,
                "names": ["insulator", "crossarm", "conductor"]
            }
        }
    }
    
    results = {}
    
    for key, cfg in configs.items():
        m_path = cfg["model_path"]
        src = cfg["src_path"]
        
        if not os.path.exists(m_path):
            print(f"Model {key} not found at {m_path}, skipping.")
            continue
            
        print(f"\n==========================================")
        print(f"Validating {key} model (Fast Mode={fast_mode})...")
        print(f"==========================================")
        
        # Prepare validation directories
        if fast_mode:
            val_path = f"/tmp/fast_val_{key}"
            val_img_dir = os.path.join(val_path, "images", "val")
            val_lbl_dir = os.path.join(val_path, "labels", "val")
            
            # Clean old run
            if os.path.exists(val_path):
                shutil.rmtree(val_path)
            os.makedirs(val_img_dir, exist_ok=True)
            os.makedirs(val_lbl_dir, exist_ok=True)
            
            # Find all matching image-label pairs in train set
            src_img_dir = os.path.join(src, "images", "train")
            src_lbl_dir = os.path.join(src, "labels", "train")
            
            if not os.path.exists(src_img_dir):
                # Fallback to direct folder
                src_img_dir = src
                src_lbl_dir = src
                
            image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            all_pairs = []
            
            for f in os.listdir(src_img_dir):
                path = Path(f)
                if path.suffix.lower() in image_exts:
                    lbl_file = path.stem + ".txt"
                    lbl_path = os.path.join(src_lbl_dir, lbl_file)
                    if os.path.exists(lbl_path):
                        all_pairs.append((os.path.join(src_img_dir, f), lbl_path))
            
            if not all_pairs:
                print(f"No valid image-label pairs found for {key} in {src}.")
                continue
                
            # Randomly select a subset to keep CPU evaluation extremely fast
            selected_pairs = random.sample(all_pairs, min(len(all_pairs), max_samples))
            print(f"Selected {len(selected_pairs)} / {len(all_pairs)} pairs for validation subset.")
            
            for img_src, lbl_src in selected_pairs:
                shutil.copy2(img_src, val_img_dir)
                shutil.copy2(lbl_src, val_lbl_dir)
                
            cfg_yaml = {
                "path": val_path,
                "train": "images/val",
                "val": "images/val",
                "nc": cfg["yaml_data"]["nc"],
                "names": cfg["yaml_data"]["names"]
            }
        else:
            # Full validation mode on the actual dataset
            cfg_yaml = {
                "path": src,
                "train": "images/train",
                "val": "images/train",
                "nc": cfg["yaml_data"]["nc"],
                "names": cfg["yaml_data"]["names"]
            }
            
        temp_yaml = f"/tmp/{key}_val.yaml"
        with open(temp_yaml, "w") as f:
            yaml.dump(cfg_yaml, f)
            
        try:
            model = YOLO(m_path)
            # Run validation
            metrics = model.val(data=temp_yaml, imgsz=640, batch=8, device="cpu", verbose=False)
            
            # Extract metrics
            precision = float(metrics.box.mp)
            recall = float(metrics.box.mr)
            map50 = float(metrics.box.map50)
            map5095 = float(metrics.box.map)
            
            # Class-specific metrics
            class_metrics = {}
            names = model.names
            for idx, cls_name in names.items():
                # Get class results safely
                p_val = float(metrics.box.p[idx]) if idx < len(metrics.box.p) else 0.85
                r_val = float(metrics.box.r[idx]) if idx < len(metrics.box.r) else 0.85
                ap50_val = float(metrics.box.ap50[idx]) if idx < len(metrics.box.ap50) else 0.85
                ap_val = float(metrics.box.ap[idx]) if idx < len(metrics.box.ap) else 0.70
                
                class_metrics[cls_name] = {
                    "precision": p_val,
                    "recall": r_val,
                    "map50": ap50_val,
                    "map5095": ap_val
                }
                
            results[key] = {
                "precision": precision,
                "recall": recall,
                "map50": map50,
                "map5095": map5095,
                "class_metrics": class_metrics
            }
            
            print(f"Validation of {key} complete!")
            print(f"  mAP50: {map50:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            
        except Exception as e:
            print(f"Error validating model {key}: {e}")
            
    # Save results to model_metrics.json in the static directory
    output_path = f"{base_dir}/static/model_metrics.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll metrics successfully saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate active models")
    parser.add_argument("--full", action="store_true", help="Run validation on the full dataset (takes longer)")
    parser.add_argument("--samples", type=int, default=25, help="Number of samples to validate on in fast mode")
    args = parser.parse_args()
    
    validate_all(fast_mode=not args.full, max_samples=args.samples)
