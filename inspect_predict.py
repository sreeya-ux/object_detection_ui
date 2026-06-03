import cv2
from pipeline import InfrastructurePipeline

pipeline_engine = InfrastructurePipeline(
    comp_model="models/best (2).pt",
    hardware_model="models/channel_12class_v2.pt",
    shed_model="models/shed_model.pt",
    insulator_model="models/insulator_model.pt"
)

img_path = "uploads/ad5df48c-27e5-45de-b304-c0a7be7cb470.jpg"
img = cv2.imread(img_path)
h_full, w_full = img.shape[:2]

# Let's run structural detection to see if any poles are found
results = list(pipeline_engine.component_model(img_path, conf=pipeline_engine.conf, iou=pipeline_engine.iou, verbose=False))
print("Detected boxes:")
for r in results:
    for box in r.boxes:
        print("Box:", box.xyxy[0].tolist(), "Conf:", box.conf[0].item(), "Class:", box.cls[0].item())
