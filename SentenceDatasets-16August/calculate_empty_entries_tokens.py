import json
import random
import math

# Load your JSON data
with open('/home/s27mhusa_hpc/Master-Thesis/SentenceDatasets-16August/Test_ner_dataset_sentence.json', 'r') as f:
    data = json.load(f)

# Ensure it's a list of entries
if not isinstance(data, list):
    data = [data]

# Separate entries
zero_entries = [entry for entry in data if all(tag == 0 for tag in entry['ner_tags'])]
non_zero_entries = [entry for entry in data if any(tag != 0 for tag in entry['ner_tags'])]

# Let x = total final size, then 0.2x = zero_entries and 0.8x = non_zero_entries => x = len(non_zero_entries) / 0.8
final_total_size = int(math.ceil(len(non_zero_entries) / 0.8))
allowed_zero_count = final_total_size - len(non_zero_entries)

# Limit zero_entries to the allowed count
sampled_zero_entries = random.sample(zero_entries, min(allowed_zero_count, len(zero_entries)))

# Combine final dataset
final_data = non_zero_entries + sampled_zero_entries
random.shuffle(final_data)

# Save result
with open('/home/s27mhusa_hpc/Master-Thesis/SentenceDatasets-16August/Test_ner_dataset_sentence_filtered.json', 'w') as f:
    json.dump(final_data, f, indent=2)

# Print summary
print(f"Original total entries: {len(data)}")
print(f"Total non-zero entries: {len(non_zero_entries)}")
print(f"Sampled zero entries (20% target): {len(sampled_zero_entries)}")
print(f"Final total entries: {len(final_data)}")
