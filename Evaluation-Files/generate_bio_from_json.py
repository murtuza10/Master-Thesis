import spacy
import json

# Load spaCy's small English model
nlp = spacy.load("en_core_web_sm")

def convert_json_to_bio(text, annotation_json):
    doc = nlp(text)
    tokens = [token.text for token in doc]
    labels = ["O"] * len(tokens)

    # Flatten all labeled 'value' fields from the nested JSON structure
    entity_spans = []

    # print(annotation_json)
    # Process each category and its entities
    for category, entries in annotation_json.items():
        for entry in entries:
            for sublabel, data in entry.items():
                value = data.get("value", "")
                if value:
                    entity_spans.append({
                        "label": sublabel,
                        "value": value
                    })

    # Loop through the values and assign BIO tags
    for entity in entity_spans:
        label = entity["label"]
        value = entity["value"]

        # Search for the value in the text and assign BIO tags
        start_idx = text.find(value)
        
        while start_idx != -1:  # Continue as long as we find the value
            # Get the token indices for the value
            end_idx = start_idx + len(value)

            # Find the corresponding tokens that match the value in the doc
            token_start = token_end = -1
            for i, token in enumerate(doc):
                if token_start == -1 and token.idx == start_idx:
                    token_start = i  # First token of the match
                if token.idx + len(token.text) == end_idx:
                    token_end = i  # Last token of the match
                    break

            # Now assign BIO tags based on token positions
            if token_start != -1 and token_end != -1:
                for i in range(token_start, token_end + 1):
                    if i == token_start:
                        labels[i] = f"B-{label}"  # Beginning of the entity
                    else:
                        labels[i] = f"I-{label}"  # Inside the entity

            # Move to the next occurrence of the value in the text
            start_idx = text.find(value, start_idx + 1)

    return tokens, labels

def generate_bio_from_json(text_file, annotations_file):

  with open(text_file, "r", encoding="utf-8") as f:
    text = f.read()
  
  with open(annotations_file, "r", encoding="utf-8") as f:
    annotation_json = json.load(f)
  tokens, labels = convert_json_to_bio(text, annotation_json)

  # Print the results
  for token, label in zip(tokens, labels):
      print(f"{token}: {label}")

  return labels


if __name__ == "__main__":
    input_file_text = "/home/s27mhusa_hpc/pilot-uc-textmining-metadata/data/Bonares/output/Text_Files_From_Inception/00bee634-47e6-490b-89ba-2464c9f09c31_inception.txt"
    input_file_annotations = "/home/s27mhusa_hpc/pilot-uc-textmining-metadata/data/Bonares/output/Results_new_prompt_json/filtered_df_soil_crop_year_LTE_test_annotated_Qwen2.5-7B-Instruct/00bee634-47e6-490b-89ba-2464c9f09c31_annotated.txt"

    generate_bio_from_json(input_file_text,input_file_annotations)