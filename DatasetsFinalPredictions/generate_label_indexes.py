import os
import zipfile
from pathlib import Path
import shutil
import os
import ast
import json



label_list = ["O","B-cropSpecies","I-cropSpecies","B-cropVariety","I-cropVariety","B-locationName","I-locationName","B-startTime","I-startTime","B-endTime","I-endTime","B-duration","I-duration","B-soilDepth","I-soilDepth","B-soilReferenceGroup","I-soilReferenceGroup","B-soilOrganicCarbon","I-soilOrganicCarbon","B-soilTexture","I-soilTexture","B-soilBulkDensity","I-soilBulkDensity","B-soilAvailableNitrogen","I-soilAvailableNitrogen","B-soilPH","I-soilPH"]


label_to_index = {label: idx for idx, label in enumerate(label_list)}

input_dir_labels = Path("/home/s27mhusa_hpc/Master-Thesis/DatasetsFinalPredictions/Test_BIO_labels_English")
output_dir_labels = Path("/home/s27mhusa_hpc/Master-Thesis/DatasetsFinalPredictions/Test_BIO_labels_indexed_English")
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
