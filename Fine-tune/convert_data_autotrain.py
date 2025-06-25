from typing import List, Dict
import json


label_map = {0: "O", 1: "B-soilOrganicCarbon",2: "I-soilOrganicCarbon",3: "B-soilTexture",4: "I-soilTexture",5: "B-startTime",6: "I-startTime",7: "B-endTime",8: "I-endTime",9: "B-city",10: "I-city",11: "B-duration",12: "I-duration",13: "B-cropSpecies",14: "I-cropSpecies",15: "B-soilAvailableNitrogen",16: "I-soilAvailableNitrogen",17: "B-soilDepth",18: "I-soilDepth",19: "B-region",20: "I-region",21: "B-country",22: "I-country",23: "B-longitude",24: "I-longitude",25: "B-latitude",26: "I-latitude",27: "B-cropVariety",28: "I-cropVariety",29: "B-soilPH",30: "I-soilPH",31: "B-soilBulkDensity",32: "I-soilBulkDensity"}


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
with open("/home/s27mhusa_hpc/Master-Thesis/combined_ner_dataset_combined_manual_val_token_minlabel.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Convert
converted_data = convert_to_span_format(raw_data)

# Save
with open("/home/s27mhusa_hpc/Master-Thesis/combined_ner_dataset_val_converted.json", "w", encoding="utf-8") as f:
    json.dump(converted_data, f, indent=2, ensure_ascii=False)

print("✅ Conversion complete! Saved to 'converted_autotrain.json'")
