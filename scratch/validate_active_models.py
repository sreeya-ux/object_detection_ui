import os
import json
import yaml
from pathlib import Path
from ultralytics import YOLO

def validate_all():
    # Define models and their configurations
    base_dir = "/home/ubuntu/object_detection_ui"
    
    configs = {
        "pole": {
            "model_path": f"{base_dir}/models/best (2).pt",
            "yaml_data": {
                "path": f"{base_dir}/training_data_component",
                "train": "images/train",
                "val": "images/train",
                "nc": 2,
                "names": {
                    0: "main_pole",
                    1: "strut_pole"
                }
            }
        },
        "components": {
            "model_path": f"{base_dir}/models/channel_12class_v2.pt",
            "yaml_data": {
                "path": f"{base_dir}/dataset_channels",
                "train": "images/train",
                "val": "images/train",
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
            "yaml_data": {
                "path": f"{base_dir}/training_data_hardware",
                "train": "images/train",
                "val": "images/train",
                "nc": 3,
                "names": ["insulator", "crossarm", "conductor"]
            }
        }
    }
    
    results = {}
    
    for key, cfg in configs.items():
        m_path = cfg["model_path"]
        if not os.path.exists(m_path):
            print(f"Model {key} not found at {m_path}, skipping.")
            continue
            
        print(f"\n==========================================")
        print(f"Validating {key} model...")
        print(f"==========================================")
        
        # Write temporary Linux yaml config
        temp_yaml = f"/tmp/{key}_val.yaml"
        with open(temp_yaml, "w") as f:
            yaml.dump(cfg["yaml_data"], f)
            
        try:
            model = YOLO(m_path)
            # Run validation with a small batch size to save memory and CPU
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
                class_metrics[cls_name] = {
                    "precision": float(metrics.box.p[idx]),
                    "recall": float(metrics.box.r[idx]),
                    "map50": float(metrics.box.ap50[idx]),
                    "map5095": float(metrics.box.ap[idx])
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
            
    # Save results to a json file in the static directory
    output_path = f"{base_dir}/static/model_metrics.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll metrics successfully saved to {output_path}")

if __name__ == "__main__":
    validate_all()
