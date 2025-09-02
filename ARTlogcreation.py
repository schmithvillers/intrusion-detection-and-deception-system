# generate_attacks.py
import os
import subprocess
import time
import yaml
from tqdm import tqdm
from datetime import datetime

# --- Configuration ---
HONEYPOT_IP = "192.168.251.150"
SSH_USER = "ubuntu_server"
SSH_PASS = "anjsudpoo"
ART_PATH = "/home/kali/redcanaryco-atomic-red-team-dffd968/atomics/"
LOCAL_HONEYPOT_LOG = "/var/log/honeypot/all_honeypot_logs.log"
LOG_TIMESTAMP_FILE = "./Logs/attack_timestamps.txt"

def run_atomic_red_team():
    """
    Clears the honeypot log, runs Atomic Red Team attacks via SSH,
    and records the start and end timestamps for log labeling.
    """
    if not os.path.exists(os.path.dirname(LOG_TIMESTAMP_FILE)):
        os.makedirs(os.path.dirname(LOG_TIMESTAMP_FILE))
        
    print("--- Preparing honeypot and local environment ---")
    
    # Reset the local log file for a clean run
    with open(LOCAL_HONEYPOT_LOG, 'w') as f:
        f.write("")
    
    # Run some benign activity to create a baseline of non-malicious logs
    print("Generating some benign logs...")
    subprocess.run(["sshpass", "-p", SSH_PASS, "ssh", f"{SSH_USER}@{HONEYPOT_IP}", "sudo apt-get update"], timeout=60, capture_output=True)
    time.sleep(5)
    
    print("\n--- Starting malicious event generation with Atomic Red Team ---")
    attack_start_time = datetime.now()
    total_commands = 0
    successful_commands = 0
    
    techniques = [d for d in os.listdir(ART_PATH) if d.startswith('T')]
    if not techniques:
        print(f"Error: No Atomic Red Team techniques found in {ART_PATH}.")
        return

    for technique in tqdm(techniques, desc="Running Atomic Tests"):
        yaml_path = os.path.join(ART_PATH, technique, f"{technique}.yaml")
        if not os.path.exists(yaml_path):
            continue

        try:
            with open(yaml_path, 'r') as file:
                atomic_data = yaml.safe_load(file)
            for test in atomic_data.get('atomic_tests', []):
                if 'linux' in test.get('supported_platforms', []):
                    command_to_run = test.get('executor', {}).get('command')
                    if command_to_run:
                        total_commands += 1
                        full_command = [
                            "sshpass", "-p", SSH_PASS, "ssh", f"{SSH_USER}@{HONEYPOT_IP}",
                            f"sh -c \"{command_to_run}\" 2>&1"
                        ]
                        try:
                            process = subprocess.run(full_command, capture_output=True, text=True, timeout=60, check=False)
                            if process.returncode == 0:
                                successful_commands += 1
                            time.sleep(0.5)
                        except subprocess.TimeoutExpired:
                            continue
        except Exception as e:
            print(f"Error processing {yaml_path}: {e}")
            
    attack_end_time = datetime.now()
    
    print("\n--- Atomic Red Team event generation complete ---")
    print(f"Summary: {successful_commands} out of {total_commands} commands ran successfully.")
    
    with open(LOG_TIMESTAMP_FILE, 'w') as f:
        f.write(f"start_time: {attack_start_time}\n")
        f.write(f"end_time: {attack_end_time}\n")
    print(f"Attack timestamps saved to {LOG_TIMESTAMP_FILE}")

if __name__ == "__main__":
    run_atomic_red_team()
