
import torch
import segmentation_models_pytorch as smp
import cv2
import numpy as np
import os
import uuid
from pipeline import InfrastructurePipeline

# Mocking the initialization logic from app.py
pipeline_engine = InfrastructurePipeline(
    component_model_path="dry_backup/best_whole.pt",
    insulator_model_path="dry_backup/best_insulator.pt",
    shed_model_path="dry_backup/best_disc.pt"
)

unet_model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
unet_model.load_state_dict(torch.load("best_cable_unet.pth", map_location="cpu"))
unet_model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
unet_model.to(device)

def test_prediction():
    test_img = "dry_backup/test.jpeg"
    if not os.path.exists(test_img):
        print(f"Error: {test_img} not found")
        return

    print(f"Testing prediction on {test_img}...")
    try:
        # Load image for processing
        img = cv2.imread(test_img)
        h, w = img.shape[:2]
        
        # 1. Pipeline
        pipe_res = pipeline_engine.predict(test_img, visualize=False)
        print("Pipeline prediction successful.")

        # 2. UNet
        input_img = cv2.resize(img, (512, 512)).transpose(2, 0, 1) / 255.0
        tensor = torch.tensor(input_img[None, ...], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            out = unet_model(tensor)
            mask = torch.sigmoid(out).squeeze().cpu().numpy()
        print("UNet prediction successful.")

    except Exception as e:
        import traceback
        print("CRASH DETECTED:")
        traceback.print_exc()

if __name__ == "__main__":
    test_prediction()
