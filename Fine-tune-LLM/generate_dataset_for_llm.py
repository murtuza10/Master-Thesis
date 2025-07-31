import requests
import json
import os
import sys
import json
import argparse
from collections import defaultdict
import json


sys.path.append(os.path.abspath('..'))

from Evaluation_Files.generate_ner_prompt_fine_tuning import generate_ner_prompts


def convert_entity_format(text, entities):
    # Final output structure
    output = {
        "Crops": [],
        "Soil": [],
        "Location": [],
        "Time Statement": []
    }

    # Label → category map
    label_category_map = {
        "cropSpecies": "Crops",
        "cropVariety": "Crops",
        "soilTexture": "Soil",
        "soilReferenceGroup": "Soil",
        "soilDepth": "Soil",
        "soilBulkDensity": "Soil",
        "soilPH": "Soil",
        "soilOrganicCarbon": "Soil",
        "soilAvailableNitrogen": "Soil",
        "country": "Location",
        "region": "Location",
        "city": "Location",
        "latitude": "Location",
        "longitude": "Location",
        "startTime": "Time Statement",
        "endTime": "Time Statement",
        "duration": "Time Statement"
    }

    for ent in entities:
        label = ent["label"]
        start = ent["start"]
        end = ent["end"]

        value = text[start:end]
        category = label_category_map.get(label)

        if category:
            entry = {label: {"value": value, "span": [start, end]}}
            output[category].append(entry)

    return output

def process_file(input_file, output_file):

    with open(input_file, "r", encoding="utf-8") as infile:
        data = json.load(infile)  # list of documents

    with open(output_file, "w", encoding="utf-8") as outfile:
        for item in data:
            text = item.get("text", "")
            entities = item.get("entities", [])
            input= generate_ner_prompts(text)

            formatted = {
                "input": input,
                "output": convert_entity_format(text, entities)
            }
            outfile.write(json.dumps(formatted) + "\n")

def main():
    
    input_file = "/home/s27mhusa_hpc/Master-Thesis/FinalDatasets-21July/Test_ner_dataset_sentence_text_entity.json"
    output_file = "/home/s27mhusa_hpc/Master-Thesis/FinalDatasets-21July/Test_ner_dataset_sentence_input_output.jsonl"
    process_file(input_file, output_file)

if __name__ == "__main__":
    main()