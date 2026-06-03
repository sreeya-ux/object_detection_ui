from ultralytics import YOLO
import os
import yaml

# =========================
# CONFIG
# =========================
DATASET_DIR = "dataset_channels"
YAML_PATH = "channels.yaml"
MODEL_VARIANT = "yolo11n-seg.pt"

def create_yaml():
    """Create the YOLO config file for training."""
    data = {
        'path': os.path.abspath(DATASET_DIR),
        'train': 'images/train',
        'val': 'images/train',
        'nc': 12,
        'names': {
            0: 'insulators',
            1: 'v_cross_arm',
            2: 'tapping_arm',
            3: 'top_cleat',
            4: 'side_arm',
            5: 't_rising',
            6: 'special_clamp',
            7: 'street_light',
            8: 'stay_set',
            9: 'box_arm',
            10: 'ab_switch',
            11: 'dtr'
        }
    }
    with open(YAML_PATH, 'w') as f:
        yaml.dump(data, f)
    print(f"✅ Created {YAML_PATH}")

def train():
    create_yaml()
    
    # Load a pretrained YOLO11 model
    print(f"🚀 Loading {MODEL_VARIANT}...")
    model = YOLO(MODEL_VARIANT)

    # Start training
    print("🔥 Starting training for Channels...")
    results = model.train(
        data=YAML_PATH,
        epochs=100,
        imgsz=1024,
        batch=4,         # Lower batch for CPU stability
        name='channel_model_v1',
        device='cpu',    # Explicitly use CPU as no NVIDIA GPU was found
        patience=20,
        save=True,
        cache=True
    )
    print("✅ Training Complete!")

if __name__ == "__main__":
    train()
