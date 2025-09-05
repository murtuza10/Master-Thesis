from typing import List, Dict
import json


label_list = ["O", "B-startTime", "I-startTime", "B-endTime", "I-endTime", "B-city", "I-city", "B-duration", "I-duration", "B-cropSpecies", "I-cropSpecies", "B-region", "I-region", "B-country", "I-country", "B-Soil", "I-Soil"]

label_map = {idx: label for idx, label in enumerate(label_list)}

def convert_to_span_format(data: List[Dict]) -> List[Dict]:
    results = []
    for item in data:
        tokens = item["tokens"]
        ner_tags = item["ner_tags"]
        text = ""
        offset = 0
        entities = []
        current_entity = None

        for token, tag_id in zip(tokens, ner_tags):
            label = label_map[tag_id]
            start = len(text)
            text += token
            end = len(text)

            if label.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {"start": start, "end": end, "label": label[2:]}
            elif label.startswith("I-") and current_entity:
                current_entity["end"] = end
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

            text += " "  # add space between tokens

        if current_entity:
            entities.append(current_entity)

        results.append({
            "text": text.strip(),
            "entities": entities
        })

    return results

# Load your data
with open("/home/s27mhusa_hpc/Master-Thesis/Dataset1stSeptemberDocumentLevel/Train_NER_dataset.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Convert
converted_data = convert_to_span_format(raw_data)

# Save
with open("/home/s27mhusa_hpc/Master-Thesis/Dataset1stSeptemberDocumentLevel/Train_ner_dataset_converted.json", "w", encoding="utf-8") as f:
    json.dump(converted_data, f, indent=2, ensure_ascii=False)

print("✅ Conversion complete! Saved to 'Train_ner_dataset_converted.json'")
