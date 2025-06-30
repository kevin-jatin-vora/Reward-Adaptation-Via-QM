import subprocess
import os

def run_script(script_name, config_path):
    print(f"Running {script_name}...")
    subprocess.run(["python", script_name, "--config", config_path])

def main():
    config_file = "config.yaml"

    # Ensure the config file exists
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"{config_file} not found. Please create it before running main.py.")

    scripts = ["QL.py", "QM.py", "SFQL.py", "SQB.py"]

    for script in scripts:
        run_script(script, config_file)

    print("All scripts completed.")

if __name__ == "__main__":
    main()
