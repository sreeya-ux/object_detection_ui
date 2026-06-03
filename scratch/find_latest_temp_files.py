import os
import glob
from datetime import datetime

def find_files():
    files = glob.glob("uploads/*")
    # Sort files by modification time
    files.sort(key=os.path.getmtime, reverse=True)
    
    print("\n--- Latest Uploaded Files in uploads/ ---")
    for f in files[:15]:
        mtime = datetime.fromtimestamp(os.path.getmtime(f))
        size = os.path.getsize(f)
        print(f"File: {f}")
        print(f"  Modified: {mtime}")
        print(f"  Size: {size} bytes")

if __name__ == "__main__":
    find_files()
