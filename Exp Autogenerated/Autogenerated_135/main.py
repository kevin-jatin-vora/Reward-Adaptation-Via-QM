import subprocess
import os

def run_script(script, config_path):
    print(f"\n🔹 Running {script}...")
    result = subprocess.run(["python", script, "--config", config_path])
    if result.returncode != 0:
        print(f"❌ Error running {script}")
    else:
        print(f"✅ Completed {script}")

def main():
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError("Missing config.yaml!")

    scripts = ["QL.py", "QM.py", "SFQL.py", "SQB.py"]
    for script in scripts:
        if os.path.exists(script):
            run_script(script, config_path)
        else:
            print(f"⚠️ {script} not found. Skipping.")

    print("\n🎉 All done!")

if __name__ == "__main__":
    main()
