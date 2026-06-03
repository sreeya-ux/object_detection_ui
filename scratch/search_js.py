with open("static/script.js", "r", encoding="utf-8") as f:
    lines = f.readlines()
    
print("--- Occurrences of pole_id or related logic in script.js ---")
for idx, line in enumerate(lines):
    if "pole_id" in line or "asset_id" in line or "RDSS" in line or "Asset ID" in line:
        print(f"Line {idx+1}: {line.strip()}")
