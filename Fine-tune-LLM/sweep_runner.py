import itertools
import os
import subprocess


# Read the Hugging Face token from the environment instead of source control.
hub_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
if not hub_token:
    raise RuntimeError(
        "Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN before running the sweep."
    )

username = os.getenv("HF_USERNAME") or os.getenv("HUGGINGFACE_USERNAME")
if not username:
    raise RuntimeError(
        "Set HF_USERNAME or HUGGINGFACE_USERNAME before running the sweep."
    )


# Sweep configuration
learning_rates = [1e-5, 3e-5, 5e-5]
batch_sizes = [2, 4]
epochs_list = [3, 5]

for lr, bs, ep in itertools.product(learning_rates, batch_sizes, epochs_list):
    run_name = f"llama3-sweep-lr{lr}-bs{bs}-ep{ep}"

    cmd = [
        "autotrain",
        "llm",
        "--train",
        "--model",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "--project_name",
        run_name,
        "--data_path",
        "/home/user/Master-Thesis/FinalDatasets-21July",
        "--text_column",
        "text",
        "--use_peft",
        "--learning_rate",
        str(lr),
        "--batch_size",
        str(bs),
        "--num_train_epochs",
        str(ep),
        "--push_to_hub",
        "--repo_id",
        f"{username}/{run_name}",
        "--token",
        hub_token,
    ]

    display_cmd = cmd.copy()
    display_cmd[-1] = "<REDACTED_HF_TOKEN>"
    print("Running:", " ".join(display_cmd))
    subprocess.run(cmd, check=True)
