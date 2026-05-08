import subprocess
import os
import sys

def run_push():
    pass_file = os.path.abspath("scratch/askpass.bat")
    with open(pass_file, "w") as f:
        f.write("@echo asakta")
    
    env = os.environ.copy()
    env["GIT_ASKPASS"] = pass_file
    env["SSH_ASKPASS"] = pass_file
    env["DISPLAY"] = ":0"
    
    # Use the discovered git path
    git_path = r"C:\Program Files\Git\bin\git.exe"
    
    print(f"Attempting push using {git_path}...")
    try:
        # We use shell=True on Windows to ensure env vars like GIT_ASKPASS are respected
        result = subprocess.run(
            [git_path, "push", "origin", "main"],
            env=env,
            capture_output=True,
            text=True
        )
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print("✅ Push Successful!")
        else:
            print(f"❌ Push Failed (Exit Code: {result.returncode})")
            
    except Exception as e:
        print(f"❌ Error during push: {e}")

if __name__ == "__main__":
    run_push()
