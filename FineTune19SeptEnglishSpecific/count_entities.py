import json
from collections import defaultdict
from datasets import Dataset, DatasetDict, concatenate_datasets
from seqeval.metrics.sequence_labeling import get_entities

# Load JSON file containing a list of records
with open("/home/s27mhusa_hpc/Master-Thesis/Dataset19September/Test_NER_dataset_English.json", "r", encoding="utf-8") as f:
    records = json.load(f)

label_list = [
    "O", "B-startTime", "I-startTime", "B-endTime", "I-endTime",
    "B-city", "I-city", "B-duration", "I-duration", "B-cropSpecies", "I-cropSpecies",
    "B-region", "I-region", "B-country", "I-country", "B-Soil", "I-Soil"
]

def extract_entities(tokens, tags, label_list):
    """Extract entities from one record based on BIO tags."""
    entities = []
    current_entity = None

    for token, tag_id in zip(tokens, tags):
        label = label_list[tag_id]

        if label.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            current_entity = {"type": label[2:], "tokens": [token]}
        elif label.startswith("I-") and current_entity and current_entity["type"] == label[2:]:
            current_entity["tokens"].append(token)
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    if current_entity:
        entities.append(current_entity)

    return entities


# Global counts
global_counts = defaultdict(int)

# Process each record
for idx, record in enumerate(records):
    tokens = record["tokens"]
    tags = record["ner_tags"]

    entities = extract_entities(tokens, tags, label_list)

    # Update global counts
    for ent in entities:
        global_counts[ent["type"]] += 1

    # Print per-record entities
    print(f"\nRecord {idx+1}:")
    for ent in entities:
        print(f"  {ent['type']}: {' '.join(ent['tokens'])}")

# Print summary
print("\n=== Global Entity Counts ===")
for etype, count in global_counts.items():
    print(f"{etype}: {count}")

# Check raw test dataset before any processing
def analyze_raw_dataset(dataset, dataset_name="Raw Dataset"):
    print(f"\n=== {dataset_name} Analysis ===")
    
    total_b_tags = 0
    seqeval_entities = 0
    problematic_sequences = 0
    
    for i, example in enumerate(dataset):
        tokens = example['tokens']
        ner_tags = example['ner_tags']
        
        # Convert to label strings
        labels = [label_list[tag_id] for tag_id in ner_tags]
        
        # Count B- tags (simple count)
        b_count = sum(1 for label in labels if label.startswith('B-'))
        total_b_tags += b_count
        
        # Count seqeval entities (strict count)
        seq_entities = len(get_entities(labels))
        seqeval_entities += seq_entities
        
        # Check for problems
        if seq_entities < b_count:
            problematic_sequences += 1
            if problematic_sequences <= 5:  # Show first 5 examples
                print(f"\nProblematic sequence {i}:")
                print(f"Tokens: {tokens}")
                print(f"Labels: {labels}")
                print(f"B- tags: {b_count}, Seqeval entities: {seq_entities}")
    
    print(f"\nSummary:")
    print(f"Total B- tags: {total_b_tags}")
    print(f"Seqeval valid entities: {seqeval_entities}")
    print(f"Rejected entities: {total_b_tags - seqeval_entities}")
    print(f"Problematic sequences: {problematic_sequences}")
    
    return total_b_tags, seqeval_entities

# Run this on your raw test dataset
analyze_raw_dataset(Dataset.from_json("/home/s27mhusa_hpc/Master-Thesis/Dataset19September/Test_NER_dataset_English.json"), "Raw Test Dataset")