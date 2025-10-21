import requests
import json
import os
import sys
import json
import argparse
from collections import defaultdict
import json


sys.path.append(os.path.abspath('..'))

def generate_ner_prompts(text: str):
        """Generate system and user prompts for LLM-based NER."""
        
        entity_types = ['cropVariety', 'soilAvailableNitrogen', 'soilBulkDensity', 'soilTexture']
        
        entity_descriptions = {
            'cropVariety': 'Specific cultivar/variety name (e.g., "Golden Delicious")',
            'soilAvailableNitrogen': 'Nitrogen is present in a soil sample that is available to plants. Please only annotate explicit mentions of the available nitrogen. Make sure it is related to the nitrogen in the soil and not in fertilizers, etc',
            'soilBulkDensity': 'The dry weight of soil divided by its volume. Please annotate the term “bulk density” if it is mentioned in a text: ',
            'soilTexture': 'Soil texture measures the proportion of sand, silt, and clay-sized particles in a soil sample. Please annotate a soil texture if it is part of a soil texture classification such as the USDA Soil Texture Classification, consisting of 12 different soil textures or the soil textures of the Bodenkundliche Kartieranleitung',
        }
        
        if entity_types:
            entity_list = "\n".join([f"- {et}: {entity_descriptions.get(et, 'Agricultural entity')}" 
                                    for et in entity_types])
        else:
            entity_list = "\n".join([f"- {et}: {desc}" for et, desc in entity_descriptions.items()])
        
        system_prompt = f"""You are an expert in agricultural Named Entity Recognition (NER).
Your task is to identify and extract specific entities from agricultural text.

Entity Types to Extract:
{entity_list}

Instructions:
1. Read the input text carefully
2. Identify all entities that match the specified types
3. Return the results as a JSON object with the following structure:
{{
    "entities": [
        {{"text": "entity text", "label": "entity_type", "start": start_index, "end": end_index}},
        ...
    ]
}}

Few-shot Examples:
"""
        
        user_prompt = f"Extract entities from the following text:\n ### Text ###{text}"
        return system_prompt, user_prompt


def convert_entity_format(text, entities):
    # Final output structure - list of dictionaries
    output = []

    for ent in entities:
        label = ent["label"]
        start = ent["start"]
        end = ent["end"]
        
        value = text[start:end]
        
        entry = {
            "text": value,
            "label": label,
            "start": start,
            "end": end
        }
        output.append(entry)

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

    input_file = "/home/s27mhusa_hpc/Master-Thesis/DatasetsFinalPredictions/NER_dataset_converted.json"
    output_file = "/home/s27mhusa_hpc/Master-Thesis/DatasetsFinalPredictions/ner_dataset_input_output.jsonl"
    process_file(input_file, output_file)

if __name__ == "__main__":
    main()