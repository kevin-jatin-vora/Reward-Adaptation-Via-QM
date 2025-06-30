import yaml
import subprocess

def run_script(script_name, config_path):
    subprocess.run(["python", script_name, "--config", config_path])

def main():
    config_path = "config.yaml"
    scripts = ["QL_RL.py", "QM.py", "SFQL.py", "SQB.py"]

    for script in scripts:
        run_script(script, config_path)

if __name__ == "__main__":
    main()
