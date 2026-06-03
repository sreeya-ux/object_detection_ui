import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from skimage.morphology import skeletonize

MODEL_PATH = r"d:\NEW_ASAKTA\dry\conductor\best_cable_unet.pth"


# ------------------ LOAD MODEL ------------------
def load_model():
    model = smp.Unet("resnet34", in_channels=3, classes=1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


# ------------------ SEGMENTATION ------------------
def get_mask(model, img):
    orig_h, orig_w = img.shape[:2]

    input_img = cv2.resize(img, (512, 512)).transpose(2, 0, 1) / 255.0
    input_tensor = torch.tensor(input_img[None, ...], dtype=torch.float32)

    with torch.no_grad():
        mask = torch.sigmoid(model(input_tensor)).cpu().numpy()[0, 0]

    mask = cv2.resize(mask, (orig_w, orig_h))
    binary = (mask > 0.5).astype(np.uint8) * 255

    return binary


# ------------------ THICKNESS + SKELETON ------------------
def compute_thickness(binary):
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    skel = (skeletonize(binary / 255) > 0).astype(np.uint8)

    thickness_values = dist[skel > 0] * 2

    if len(thickness_values) == 0:
        return None, skel

    return {
        "avg": float(np.mean(thickness_values)),
        "max": float(np.max(thickness_values))
    }, skel


# ------------------ ANGLE ------------------
def compute_angle(skel):
    # np.where returns (row, col) which is (y, x). We swap to (x, y) for OpenCV
    points = np.column_stack(np.where(skel > 0)[::-1])

    if len(points) < 10:
        return None

    vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)

    angle = np.degrees(np.arctan2(vy, vx)).item()

    return float(angle)


# ------------------ POLE LEAN ------------------
def detect_pole_lean(angle):
    if angle is None:
        return "No pole detected"

    deviation = abs(90 - abs(angle))

    if deviation > 10:
        return f"FAULT: Pole Leaning ({deviation:.2f}°)"
    else:
        return "Pole Normal"


# ------------------ CONDUCTOR SAG ------------------
def detect_conductor_sag(skel):
    # np.where returns (row, col) which is (y, x). We swap to (x, y)
    points = np.column_stack(np.where(skel > 0)[::-1])

    if len(points) < 20:
        return "No conductor detected"

    # sort points along x-axis (index 0 now)
    points = points[np.argsort(points[:, 0])]

    x = points[:, 0]
    y = points[:, 1]

    # fit straight line
    coeffs = np.polyfit(x, y, 1)
    y_fit = np.polyval(coeffs, x)

    # deviation from straight line
    deviation = np.mean(np.abs(y - y_fit))

    if deviation > 5:
        return f"FAULT: Conductor Sagging"
    else:
        return "Conductor Normal"


# ------------------ MAIN ------------------
def measure(img_path):
    model = load_model()

    img = cv2.imread(img_path)
    if img is None:
        print("Error: Image not found")
        return

    binary = get_mask(model, img)

    thickness, skel = compute_thickness(binary)

    # ---- ANGLE (used for pole approx) ----
    angle = compute_angle(skel)

    # ---- DETECTIONS ----
    pole_status = detect_pole_lean(angle)
    conductor_status = detect_conductor_sag(skel)

    # ---- OUTPUT ----
    print("\n--- RESULTS ---")

    if thickness:
        print(f"Thickness Avg: {thickness['avg']:.2f}px")
        print(f"Thickness Max: {thickness['max']:.2f}px")
    else:
        print("No thickness detected")

    if angle is not None:
        print(f"Angle: {angle:.2f} degrees")
    else:
        print("Angle could not be computed")

    print(f"Pole Status: {pole_status}")
    print(f"Conductor Status: {conductor_status}")


# ------------------ ENTRY ------------------
if __name__ == "__main__":
    import sys
    measure(sys.argv[1])
