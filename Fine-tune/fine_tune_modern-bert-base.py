import subprocess
import itertools
import os
import re
import csv

# Sweep values
learning_rates = [1e-5, 3e-5, 5e-5]
batch_sizes = [2, 4]
epochs_list = [3, 5]

# Hugging Face Hub credentials
username = "murtuza10"
hub_token = "hf_your_token_here"  # ← Replace with your actual token

# Project prefix
base_project = "llama3-sweep"

# Log and results setup
os.makedirs("logs", exist_ok=True)
log_file_path = "logs/sweep_log.txt"
results_csv_path = "logs/results.csv"

# Open CSV writer
with open(results_csv_path, "w", newline="") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["Run Name", "Learning Rate", "Batch Size", "Epochs", "Eval Loss"])

    # Run sweep combinations
    for lr, bs, ep in itertools.product(learning_rates, batch_sizes, epochs_list):
        run_name = f"{base_project}-lr{lr}-bs{bs}-ep{ep}"
        print(f"\n🚀 Starting: {run_name}")

        cmd = [
            "autotrain", "llm",
            "--config", "llm_config.yaml",
            "--train",
            "--project-name", run_name,
            "--lr", str(lr),
            "--epochs", str(ep),
            "--batch-size", str(bs),
            "--push-to-hub",
            "--hub-token", hub_token,
            "--hub-username", username
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        stdout = result.stdout + result.stderr

        # Log to text file
        with open(log_file_path, "a") as log_file:
            log_file.write(f"=== Run: {run_name} ===\n")
            log_file.write(f"LR: {lr}, Batch Size: {bs}, Epochs: {ep}\n")
            log_file.write(stdout + "\n\n")

        # Extract eval loss (looks for line like: "eval_loss = 1.234")
        match = re.search(r"eval_loss\s*=\s*([0-9.]+)", stdout)
        eval_loss = float(match.group(1)) if match else "N/A"

        # Write to CSV
        csvwriter.writerow([run_name, lr, bs, ep, eval_loss])

        print(f"✅ Finished: {run_name} — Eval Loss: {eval_loss}")
