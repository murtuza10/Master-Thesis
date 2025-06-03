import spacy
import json

# Load spaCy's small English model
nlp = spacy.load("en_core_web_sm")

def convert_json_to_bio(text, annotation_json):
    doc = nlp(text)
    tokens = [token.text for token in doc]
    labels = ["O"] * len(tokens)

    entity_spans = []

    # Flatten all labeled 'value' fields from the nested JSON structure
    for category, entries in annotation_json.items():
        for entry in entries:
            for sublabel, data in entry.items():
                value = data.get("value", "")
                span = data.get("span", [])
                if value:
                    entity_spans.append({
                        "label": sublabel,
                        "value": value,
                        "span": span
                    })

    correct_spans = 0
    fallbacks_used = 0
    not_found_entities_count = 0
    not_found_entities = []

    for entity in entity_spans:
        label = entity["label"]
        value = entity["value"]
        span = entity.get("span", [])

        start_idx = -1

        # Check if the span is valid and matches the value
        if isinstance(span, list) and len(span) == 2 and all(isinstance(x, int) for x in span):
            start, end = span
            if text[start:end] == value:
                start_idx = start
                correct_spans += 1
            else:
                start_idx = text.find(value)
                if start_idx != -1:
                    fallbacks_used += 1
                else:
                    not_found_entities_count += 1
                    not_found_entities.append(value)
                    continue  # Skip tagging
        else:
            # Span is not usable; try finding the value in text
            start_idx = text.find(value)
            if start_idx != -1:
                fallbacks_used += 1
            else:
                not_found_entities_count += 1
                not_found_entities.append(value)
                continue  # Skip tagging

        end_idx = start_idx + len(value)

        # Find token positions
        token_start = token_end = -1
        for i, token in enumerate(doc):
            if token_start == -1 and token.idx == start_idx:
                token_start = i
            if token.idx + len(token.text) == end_idx:
                token_end = i
                break

        if token_start != -1 and token_end != -1:
            for i in range(token_start, token_end + 1):
                labels[i] = f"B-{label}" if i == token_start else f"I-{label}"

    stats = {
        "correct_spans": correct_spans,
        "fallbacks_used": fallbacks_used,
        "not_found_entities": not_found_entities,
        "not_found_entities_count": not_found_entities_count,
        "total_entities": len(entity_spans)
    }

    return tokens, labels, stats

def generate_bio_from_json(text_file, annotations_file):

  with open(text_file, "r", encoding="utf-8") as f:
    text = f.read()
  
  with open(annotations_file, "r", encoding="utf-8") as f:
    annotation_json = json.load(f)
  tokens, labels, stats = convert_json_to_bio(text, annotation_json)

#   Print the results
#   for token, label in zip(tokens, labels):
#       print(f"{token}: {label}")
  print(stats)

  return tokens, labels, stats


if __name__ == "__main__":
    input_file_text = "/home/s27mhusa_hpc/Master-Thesis/Text_Files_For_LLM_Input/00bee634-47e6-490b-89ba-2464c9f09c31_inception.txt"
    input_file_annotations = "/home/s27mhusa_hpc/Master-Thesis/Results/Results_new_prompt_json/LLM_annotated_Qwen2.5-72B-Instruct/00bee634-47e6-490b-89ba-2464c9f09c31_inception_annotated.txt"

    generate_bio_from_json(input_file_text,input_file_annotations)