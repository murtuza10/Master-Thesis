import os
import zipfile
from pathlib import Path
import shutil
import os
import ast
import json



label_list = ["O","B-Crop", "I-Crop", "B-TimeStatement","I-TimeStatement","B-Location","I-Location"]


label_to_index = {label: idx for idx, label in enumerate(label_list)}

input_dir_labels = Path("/home/s27mhusa_hpc/Master-Thesis/Dataset19SeptemberNoSoil/Train_BIO_labels_Broad_NoSoil_sentence")
output_dir_labels = Path("/home/s27mhusa_hpc/Master-Thesis/Dataset19SeptemberNoSoil/Train_BIO_labels_Broad_NoSoil_sentence_indexed")
os.makedirs(output_dir_labels, exist_ok=True)

for filename in os.listdir(input_dir_labels):
    if not filename.endswith(".txt") and not filename.endswith(".json"):
        continue

    input_path = os.path.join(input_dir_labels, filename)
    output_path = os.path.join(output_dir_labels, filename)

    with open(input_path, "r", encoding="utf-8") as infile:
        try:
            content = infile.read().strip()
            labels = ast.literal_eval(content)  # safely parse the list
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

    try:
        indices = [label_to_index[label] for label in labels]
    except KeyError as e:
        print(f"Label not found in label_list: {e} in file {filename}")
        continue

    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(indices, outfile)
