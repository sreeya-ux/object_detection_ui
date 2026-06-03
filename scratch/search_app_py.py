with open("app.py", "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        if "_patch" in line:
            print(f"Line {idx+1}: {line.strip()}")
