
from ultralytics import YOLO
import cv2

def debug_model(model_path, image_path):
    print(f"DEBUG: Loading model {model_path}...")
    model = YOLO(model_path)
    
    print(f"DEBUG: Running inference on {image_path} with ultra-low threshold (0.01)...")
    results = model.predict(image_path, conf=0.01, task='obb')
    
    print("\n--- RAW DETECTIONS ---")
    for r in results:
        if r.obb:
            print(f"Found {len(r.obb)} OBB boxes:")
            for box in r.obb:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names[cls]
                print(f"  - [{name}] conf: {conf:.4f}")
        else:
            print("No OBB boxes found. Let's try standard detection pass...")
            res_det = model.predict(image_path, conf=0.01, task='detect')
            for r_d in res_det:
                if r_d.boxes:
                    print(f"Found {len(r_d.boxes)} standard boxes:")
                    for b in r_d.boxes:
                        cls = int(b.cls[0])
                        conf = float(b.conf[0])
                        print(f"  - [{model.names[cls]}] conf: {conf:.4f}")
                else:
                    print("No detections found at all (even at 1% confidence).")

if __name__ == "__main__":
    debug_model(r"c:\Users\ASK037-PC\Downloads\object_detection_ui\best (2)\best (2).pt", 
                r"c:\Users\ASK037-PC\.gemini\antigravity\brain\8b4adf17-a991-49b2-ae58-0a654dd7c2df\media__1778150808440.jpg")
