import traceback
from app import process_image_file
class DummyFile:
    def read(self):
        with open("test_result_final.jpg", "rb") as f:
            return f.read()

try:
    process_image_file(DummyFile())
    print("SUCCESS")
except Exception as e:
    traceback.print_exc()
