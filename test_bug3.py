import json
from app import process_image_file
class DummyFile:
    def read(self):
        with open("test_result_final.jpg", "rb") as f:
            return f.read()

try:
    res = process_image_file(DummyFile())
    print("SUCCESS")
    print("Detections found:", len(res["detections"]))
    for d in res["detections"]:
        print(f" - {d['label']}, box: {d['bbox']}")
except Exception as e:
    import traceback
    traceback.print_exc()
