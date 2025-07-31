import os
import ast
import json
from pathlib import Path



# Define paths
tokens_path = Path("/home/s27mhusa_hpc/Master-Thesis/FinalDatasets-10July/Test_BIO_tokens")
labels_path = Path("/home/s27mhusa_hpc/Master-Thesis/FinalDatasets-10July/Test_BIO_labels_indexed")
output_path = Path("/home/s27mhusa_hpc/Master-Thesis/FinalDatasets-10July/Test_ner_dataset_sentence.json")


def split_tokens_and_labels(tokens, labels, sentence_end_tokens={".", "!", "?"}):
    """Split tokens and labels into sentence-level chunks."""
    sentence_tokens = []
    sentence_labels = []
    all_sentences = []

    for token, label in zip(tokens, labels):
        sentence_tokens.append(token)
        sentence_labels.append(label)
        if token in sentence_end_tokens:
            all_sentences.append({
                "tokens": sentence_tokens,
                "ner_tags": sentence_labels
            })
            sentence_tokens = []
            sentence_labels = []

    if sentence_tokens:  # leftover sentence
        all_sentences.append({
            "tokens": sentence_tokens,
            "ner_tags": sentence_labels
        })

    return all_sentences

all_data = []

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

    # Split and accumulate
    sentence_data = split_tokens_and_labels(tokens, ner_tags)
    all_data.extend(sentence_data)

# Write to JSON
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(all_data, f, indent=2, ensure_ascii=False)

print(f"✅ Saved {len(all_data)} sentences to {output_path}")