import json
import random

# File paths
input_path = '/home/s27mhusa_hpc/Master-Thesis/SentenceDatasets6thAugust/Combined_ner_dataset_sentence_chat_final.jsonl'
output_path = '/home/s27mhusa_hpc/Master-Thesis/SentenceDatasets6thAugust/Combined_ner_dataset_sentence_chat_final_filtered.jsonl'

# Data containers
empty_entries = []
non_empty_entries = []

# Read and classify entries
with open(input_path, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data = json.loads(line)
            assistant_reply = next((item["content"] for item in data["messages"] if item["role"] == "assistant"), None)
            if assistant_reply:
                entities = json.loads(assistant_reply)
                if all(not entities.get(cat) for cat in ["Crops", "Soil", "Location", "Time Statement"]):
                    empty_entries.append(data)
                else:
                    non_empty_entries.append(data)
            else:
                non_empty_entries.append(data)
        except Exception as e:
            print(f"⚠️ Skipping malformed entry: {e}")
            continue

# Correct logic: limit empty to 25% of non-empty entries
allowed_empty = min(len(empty_entries), int(0.25 * len(non_empty_entries)))
selected_empty = random.sample(empty_entries, allowed_empty)

# Combine and shuffle
final_entries = non_empty_entries + selected_empty
random.shuffle(final_entries)

# Write output
with open(output_path, 'w', encoding='utf-8') as f:
    for entry in final_entries:
        f.write(json.dumps(entry) + '\n')

print(f"✅ Total original entries: {len(empty_entries) + len(non_empty_entries)}")
print(f"💡 Non-empty entries kept: {len(non_empty_entries)}")
print(f"🧹 Empty entries reduced to: {len(selected_empty)} (max 20% of final set)")
print(f"📁 Final entries written to: {output_path}")