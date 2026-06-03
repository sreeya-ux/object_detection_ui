def run_test():
    log_path = "output.log"
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading log: {e}")
        return
        
    # Find indices matching POST /predict
    matches = []
    for idx, line in enumerate(lines):
        if "POST /predict" in line:
            matches.append(idx)
            
    print(f"Found {len(matches)} POST /predict requests.")
    for m_idx in matches:
        print(f"\n==================================================")
        print(f"POST /predict found at line {m_idx+1}:")
        print(lines[m_idx].strip())
        print("==================================================")
        
        # Print 50 lines before and 20 lines after
        start = max(0, m_idx - 60)
        end = min(len(lines), m_idx + 10)
        for i in range(start, end):
            prefix = ">>>" if i == m_idx else f"{i+1:3d}"
            print(f"{prefix}: {lines[i].strip()}")

if __name__ == "__main__":
    run_test()
