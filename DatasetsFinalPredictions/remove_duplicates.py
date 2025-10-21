import json

def remove_duplicates_from_json(input_file, output_file):
    """
    Remove duplicate entries from a JSON file containing tokenized data.
    Duplicates are identified by identical token sequences.
    """
    # Read the JSON file
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Track unique entries using token sequences as keys
    seen_tokens = set()
    unique_entries = []
    
    for entry in data:
        # Convert tokens list to tuple for hashing
        tokens_tuple = tuple(entry['tokens'])
        
        # If this token sequence hasn't been seen before, keep it
        if tokens_tuple not in seen_tokens:
            seen_tokens.add(tokens_tuple)
            unique_entries.append(entry)
    
    # Write the deduplicated data to output file
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(unique_entries, file, indent=2, ensure_ascii=False)
    
    print(f"Original entries: {len(data)}")
    print(f"Unique entries: {len(unique_entries)}")
    print(f"Duplicates removed: {len(data) - len(unique_entries)}")

# Alternative version that also considers NER tags for stricter duplicate detection
def remove_duplicates_strict(input_file, output_file):
    """
    Remove duplicates considering both tokens and NER tags.
    """
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    seen_entries = set()
    unique_entries = []
    
    for entry in data:
        # Create a hash from both tokens and NER tags
        entry_key = (tuple(entry['tokens']), tuple(entry['ner_tags']))
        
        if entry_key not in seen_entries:
            seen_entries.add(entry_key)
            unique_entries.append(entry)
    
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(unique_entries, file, indent=2, ensure_ascii=False)
    
    print(f"Original entries: {len(data)}")
    print(f"Unique entries: {len(unique_entries)}")
    print(f"Duplicates removed: {len(data) - len(unique_entries)}")

# Usage examples
if __name__ == "__main__":
    # Remove duplicates based on tokens only
    remove_duplicates_from_json('/home/s27mhusa_hpc/Master-Thesis/DatasetsFinalPredictions/NER_dataset_filtered.json', '/home/s27mhusa_hpc/Master-Thesis/DatasetsFinalPredictions/NER_dataset_filtered_nodupl.json')

    # Or use strict mode (considering both tokens and NER tags)
    # remove_duplicates_strict('/home/s27mhusa_hpc/Master-Thesis/Dataset19September/Train_NER_dataset_filtered.json', '/home/s27mhusa_hpc/Master-Thesis/Dataset19September/output_strict_deduplicated.json')
