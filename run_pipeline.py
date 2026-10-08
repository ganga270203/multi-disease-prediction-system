import subprocess
import sys
import os

def run(script):
    print(f"\n{'='*50}")
    print(f"RUNNING: {script}")
    print("="*50)
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"[ERROR] {script} failed.")
        sys.exit(1)

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run("src/train_all.py")
    print("\n" + "="*50)
    print("✅ ALL MODELS TRAINED!")
    print("Run the app: streamlit run app.py")
    print("="*50)