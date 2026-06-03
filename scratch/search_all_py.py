import glob

files = glob.glob("*.py")
for f in files:
    with open(f, "r", encoding="utf-8", errors="ignore") as file:
        for idx, line in enumerate(file):
            if "_patch" in line:
                print(f"File {f}, Line {idx+1}: {line.strip()}")
