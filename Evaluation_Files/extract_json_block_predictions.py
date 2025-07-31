import json
import re

input_file = '/home/s27mhusa_hpc/Master-Thesis/FinalDatasets-21July/Test_predictions_Pretrained_Qwen2.5-72B.jsonl'
output_file = '/home/s27mhusa_hpc/Master-Thesis/FinalDatasets-21July/Test_predictions_Pretrained_Qwen2.5-72B_extracted.json'


extracted_json_objects = []

# Regex to extract JSON block from triple backticks
json_block_pattern = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

with open(input_file, 'r', encoding='utf-8') as infile:
    for line in infile:
        try:
            record = json.loads(line)
            prediction = record.get("prediction", "")
            record_id = record.get("id")

            match = json_block_pattern.search(prediction)
            if match:
                json_str = match.group(1)
                entities = json.loads(json_str)
                extracted_json_objects.append({
                    "id": record_id,
                    "entities": entities
                })
            else:
                print(f"No JSON block found in prediction for id: {record_id}")

        except json.JSONDecodeError as e:
            print(f"Skipping malformed line: {e}")

# Save all results
with open(output_file, 'w', encoding='utf-8') as outfile:
    json.dump(extracted_json_objects, outfile, indent=2)