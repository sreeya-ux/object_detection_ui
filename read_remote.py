import sys

path = '/home/ubuntu/object_detection_ui/templates/admin.html'
try:
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    print("Total lines:", len(lines))
    print("--- LINES 90 to 140 ---")
    for i in range(90, min(140, len(lines))):
        print(f"{i+1}: {lines[i]}", end='')
except Exception as e:
    print("Error:", e)
