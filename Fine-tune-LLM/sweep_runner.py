

import subprocess
import itertools
import os

# Verify token exists
hub_token = "hf_4d6F51WHf61AjoGx1cqqFE6ksHtCcir4gjdg1guX5mcV"  


# Sweep configuration
learning_rates = [1e-5, 3e-5, 5e-5]
batch_sizes = [2, 4]
epochs_list = [3, 5]
username = "murtuza10"

for lr, bs, ep in itertools.product(learning_rates, batch_sizes, epochs_list):
    run_name = f"llama3-sweep-lr{lr}-bs{bs}-ep{ep}"
    
    cmd = [
        "autotrain", "llm",
        "--train",
        "--model", "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "--project_name", run_name,
        "--data_path", "/home/s27mhusa_hpc/Master-Thesis/FinalDatasets-21July",
        "--text_column", "text",
        "--use_peft",
        "--learning_rate", str(lr),
        "--batch_size", str(bs),
        "--num_train_epochs", str(ep),
        "--push_to_hub",
        "--repo_id", f"{username}/{run_name}",
        "--token", hub_token  # Note: --token instead of --hub-token
    ]
    
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)