import json
NDJSON_PATH = r"C:\Users\ASK037-PC\Downloads\Export  project - channels,insutaors,conductors - 5_8_2026 (4).ndjson"

with open(NDJSON_PATH, "r") as f:
    for line in f:
        if not line.strip(): continue
        item = json.loads(line)
        for p_id in item["projects"]:
            labels = item["projects"][p_id].get("labels", [])
            if labels:
                objs = labels[0]["annotations"]["objects"]
                if objs:
                    print(f"Project: {p_id}")
                    for o in objs:
                        print(f" - Value: '{o.get('value')}', Name: '{o.get('name')}'")
                    import sys
                    sys.exit(0)
