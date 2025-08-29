import json

# File paths
input_file = '/home/s27mhusa_hpc/Master-Thesis/LLM-Predictions-Document-8Aug/Test_gold_Llama-3.1-8B-Instruct-Document-1.jsonl'
output_file = '/home/s27mhusa_hpc/Master-Thesis/LLM-Predictions-Document-8Aug/Test_gold_Llama-3.1-8B-Instruct-Document-1_extracted.json'

# List to store id + gold_output entries
extracted_entries = []

# Read each line and extract id + gold_output
with open(input_file, 'r') as infile:
    for line in infile:
        data = json.loads(line)
        if "gold_output" in data and "id" in data:
            extracted_entries.append({
                "id": data["id"],
                "gold_output": data["gold_output"]
            })

# Save the extracted entries to a new JSON file
with open(output_file, 'w') as outfile:
    json.dump(extracted_entries, outfile, indent=4)

print(f"Extracted {len(extracted_entries)} entries with IDs to {output_file}")
