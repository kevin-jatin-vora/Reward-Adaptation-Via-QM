import subprocess
import os

def run_script(script_name, config_path):
    print(f"\n🔹 Running {script_name}...")
    result = subprocess.run(["python", script_name, "--config", config_path])
    if result.returncode != 0:
        print(f"❌ Failed: {script_name}")
    else:
        print(f"✅ Finished: {script_name}")

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

    print("\n🎉 All experiments complete.")

if __name__ == "__main__":
    main()
