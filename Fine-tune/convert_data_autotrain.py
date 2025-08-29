from typing import List, Dict
import json


label_map = {0: "O",1:"B-soilReferenceGroup",2:"I-soilReferenceGroup", 3: "B-soilOrganicCarbon",4: "I-soilOrganicCarbon",5: "B-soilTexture",6: "I-soilTexture",7: "B-startTime",8: "I-startTime",9: "B-endTime",10: "I-endTime",11: "B-city",12: "I-city",13: "B-duration",14: "I-duration",15: "B-cropSpecies",16: "I-cropSpecies",17: "B-soilAvailableNitrogen",18: "I-soilAvailableNitrogen",19: "B-soilDepth",20: "I-soilDepth",21: "B-region",22: "I-region",23: "B-country",24: "I-country",25: "B-longitude",26: "I-longitude",27: "B-latitude",28: "I-latitude",29: "B-cropVariety",30: "I-cropVariety",31: "B-soilPH",32: "I-soilPH",33: "B-soilBulkDensity",34: "I-soilBulkDensity"}


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
with open("/home/s27mhusa_hpc/Master-Thesis/NewDatasets27August/Test_ner_dataset_sentence.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Convert
converted_data = convert_to_span_format(raw_data)

# Save
with open("/home/s27mhusa_hpc/Master-Thesis/NewDatasets27August/Test_ner_dataset_sentence_converted.json", "w", encoding="utf-8") as f:
    json.dump(converted_data, f, indent=2, ensure_ascii=False)

print("✅ Conversion complete! Saved to 'Combined_ner_dataset_document_converted.json'")
