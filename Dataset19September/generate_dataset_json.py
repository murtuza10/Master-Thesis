import os
import ast
import json
from pathlib import Path

# Define paths
tokens_path = Path("/home/s27mhusa_hpc/Master-Thesis/Dataset19September/Test_BIO_tokens_Specific_sentence")
labels_path = Path("/home/s27mhusa_hpc/Master-Thesis/Dataset19September/Test_BIO_labels_Specific_indexed_sentence")
output_path = Path("/home/s27mhusa_hpc/Master-Thesis/Dataset19September/Test_NER_dataset_Specific.json")

data = []

# Assume matching file names in both directories
for token_file in sorted(tokens_path.iterdir()):
    label_file = labels_path / token_file.name.replace("tokens_", "")
    if not label_file.exists():
        print(f"Label file missing for {token_file.name}")
        continue

    # Read token list
    with open(token_file, "r", encoding="utf-8") as tf:
        try:
            tokens = ast.literal_eval(tf.read().strip())
        except Exception as e:
            print(f"Error reading tokens from {token_file.name}: {e}")
            continue

    # Read label indices
    with open(label_file, "r", encoding="utf-8") as lf:
        try:
            ner_tags = ast.literal_eval(lf.read().strip())
        except Exception as e:
            print(f"Error reading labels from {label_file.name}: {e}")
            continue

    # Sanity check
    if len(tokens) != len(ner_tags):
        print(f"Length mismatch in {token_file.name}: {len(tokens)} tokens vs {len(ner_tags)} labels")
        continue

    data.append({
        "tokens": tokens,
        "ner_tags": ner_tags
    })

# Save as JSON
with open(output_path, "w", encoding="utf-8") as out_json:
    json.dump(data, out_json, indent=2, ensure_ascii=False)

print(f"Saved combined dataset to {output_path}")
