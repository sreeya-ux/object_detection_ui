from ultralytics import YOLO
import os

def train_hardware():
    # Load a pretrained YOLOv8s model
    model = YOLO("yolov8s.pt") 
    
    # Path to the data.yaml for hardware
    data_yaml = os.path.abspath("training_data_hardware/data.yaml")
    
    print(f"Starting HARDWARE ONLY training with data: {data_yaml}")
    
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=100,
        imgsz=640,
        batch=16,
        name="hardware_model_v1",
        project="runs/detect"
    )
    
    print("Hardware Training Complete!")
    print(f"Best hardware model saved at: {results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    train_hardware()
