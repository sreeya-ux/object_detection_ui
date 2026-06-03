import os

for root, dirs, files in os.walk('.'):
    root_parts = root.lower().replace('\\', '/').split('/')
    if any(x in root_parts for x in ['venv', '.git', '__pycache__', 'runs']):
        continue
    
    txt_files = [f for f in files if f.endswith('.txt') and f not in ['classes.txt', 'classes_main.txt']]
    if txt_files:
        print(f"Path: {root}")
        print(f"  Labels count: {len(txt_files)}")
        # Check first 5 lines of first txt file to see class IDs
        cids = set()
        for f in txt_files[:50]:
            try:
                with open(os.path.join(root, f)) as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if parts:
                            cids.add(int(parts[0]))
            except:
                pass
        print(f"  Class IDs: {sorted(list(cids))}")
