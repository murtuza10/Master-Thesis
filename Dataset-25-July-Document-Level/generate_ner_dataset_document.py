import os
import ast
import json
from pathlib import Path

# Define paths
# Ensure these paths are correct for your environment.
# Example:
# tokens_path = Path("/path/to/your/Test_BIO_tokens")
# labels_path = Path("/path/to/your/Test_BIO_labels_indexed")
# output_path = Path("/path/to/your/Test_ner_dataset_document.json")

tokens_path = Path("/home/s27mhusa_hpc/Master-Thesis/Dataset-25-July-Document-Level/Combined_BIO_tokens")
labels_path = Path("/home/s27mhusa_hpc/Master-Thesis/Dataset-25-July-Document-Level/Combined_BIO_labels_indexed")
output_path = Path("/home/s27mhusa_hpc/Master-Thesis/Dataset-25-July-Document-Level/Combined_ner_dataset_document.json") # Changed output filename to reflect document-level processing

all_data = [] # This list will store data for each complete document

# Iterate through each token file in the specified directory
for token_file in sorted(tokens_path.iterdir()):
    # Construct the corresponding label file path
    label_file = labels_path / token_file.name.replace("tokens_", "")

    # Check if the label file exists for the current token file
    if not label_file.exists():
        print(f"Label file missing for {token_file.name}. Skipping.")
        continue

    # Read token list from the token file
    with open(token_file, "r", encoding="utf-8") as tf:
        try:
            # Safely evaluate the string content as a Python literal (list of tokens)
            tokens = ast.literal_eval(tf.read().strip())
        except Exception as e:
            print(f"Error reading tokens from {token_file.name}: {e}. Skipping.")
            continue

    # Read label indices from the label file
    with open(label_file, "r", encoding="utf-8") as lf:
        try:
            # Safely evaluate the string content as a Python literal (list of labels/NER tags)
            ner_tags = ast.literal_eval(lf.read().strip())
        except Exception as e:
            print(f"Error reading labels from {label_file.name}: {e}. Skipping.")
            continue

    # Perform a sanity check to ensure token and label lists have the same length
    if len(tokens) != len(ner_tags):
        print(f"Length mismatch in {token_file.name}: {len(tokens)} tokens vs {len(ner_tags)} labels. Skipping.")
        continue

    # Instead of splitting into sentences, we now treat the entire document
    # (its tokens and corresponding NER tags) as a single entry.
    all_data.append({
        "tokens": tokens,
        "ner_tags": ner_tags
    })

# Write the accumulated data (each entry representing a full document) to a JSON file
with open(output_path, "w", encoding="utf-8") as f:
    # Use json.dump to write the list of dictionaries to the file
    # indent=2 makes the output human-readable with 2-space indentation
    # ensure_ascii=False allows non-ASCII characters (like special symbols) to be written directly
    json.dump(all_data, f, indent=2, ensure_ascii=False)

print(f"✅ Successfully processed {len(all_data)} documents and saved to {output_path}")
