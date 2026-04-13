import json
from app import process_image_file
class DummyFile:
    def read(self):
        with open("test_result_final.jpg", "rb") as f:
            return f.read()

with open("test_res.txt", "w", encoding="utf-8") as out:
    try:
        res = process_image_file(DummyFile())
        out.write("SUCCESS\n")
        out.write(f"Detections found: {len(res['detections'])}\n")
        for d in res["detections"]:
            out.write(f" - {d['label']}, box: {d['bbox']}\n")
    except Exception as e:
        import traceback
        out.write(traceback.format_exc())
