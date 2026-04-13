import traceback
from app import process_image_file
class DummyFile:
    def read(self):
        with open("test_result_final.jpg", "rb") as f:
            return f.read()

with open("test_trace.txt", "w", encoding="utf-8") as f:
    try:
        process_image_file(DummyFile())
        f.write("SUCCESS\n")
    except Exception as e:
        f.write(traceback.format_exc())
