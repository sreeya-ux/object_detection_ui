from ultralytics import YOLO
import sys

model_path = r"C:\Users\asus\Downloads\runs\content\runs\detect\insulator_model\weights\best.pt"
try:
    model = YOLO(model_path)
    print(f"Model classes: {model.names}")
except Exception as e:
    print(f"Error loading model: {e}")
