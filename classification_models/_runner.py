import subprocess
import time

def run_script(script,script_name):
    """Run a Python script and log its progress."""
    print(f"Running {script_name}...")
    process = subprocess.Popen(['python3', script])
    process.wait()  # Wait for the process to complete
    return process.returncode

# List of scripts to run and their corresponding output files
scripts = [
    # ("resnet18","resnet18.py"),
    # ("resnet34","resnet34.py"),
    # ("resnet50","resnet50.py"),
    # ("alexnet","alexnet.py"),
    ("efficientnet","efficientnet.py"),
    ("maxvit","maxvit.py"),
    ("vit","vit.py"),
    ("tocsv","tocsv.py"),
    ("metric","metric.py"),
    ("predictions_final","predictions_final.py"),
]

for script_name,script in scripts:
    result = run_script(script,script_name)
    if result == 0:
        print(f"\n\n\n{script_name} finished successfully.\n\n\n")
    else:
        print(f"Error running {script_name}.")
    
    # Wait a bit between runs to avoid overloading the system
    time.sleep(1)
