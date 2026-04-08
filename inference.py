# d:\NEW_ASAKTA\dry\conductor\inference.py
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp

MODEL_PATH = r"d:\NEW_ASAKTA\dry\conductor\best_cable_unet.pth"

def measure(img_path):
    # 1. Load Model
    model = smp.Unet("resnet34", in_channels=3, classes=1)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # 2. Run Inference
    img = cv2.imread(img_path)
    orig_h, orig_w = img.shape[:2]
    input_img = cv2.resize(img, (512, 512)).transpose(2, 0, 1) / 255.0
    input_tensor = torch.tensor(input_img[None, ...], dtype=torch.float32)
    
    with torch.no_grad():
        mask = torch.sigmoid(model(input_tensor)).cpu().numpy()[0, 0]
    
    mask = cv2.resize(mask, (orig_w, orig_h))
    binary = (mask > 0.5).astype(np.uint8) * 255

    # 3. Thickness via Distance Transform
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    
    # Skeletonize to sample thickness center-line
    from skimage.morphology import skeletonize
    skel = (skeletonize(binary/255) > 0).astype(np.uint8)
    
    thickness_values = dist[skel > 0] * 2
    if len(thickness_values) > 0:
        print(f"Cable Thickness (Pixel Avg): {np.mean(thickness_values):.2f}")
        print(f"Cable Thickness (Max): {np.max(thickness_values):.2f}")
    else:
        print("No cable detected.")

if __name__ == "__main__":
    import sys
    measure(sys.argv[1])
