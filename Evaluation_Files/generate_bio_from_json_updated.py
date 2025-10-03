import spacy
import json

# Load spaCy's small English model
nlp = spacy.load("en_core_web_sm")

def convert_json_to_bio(text, annotation_json, nlp=None):
    """
    Convert JSON annotations to BIO format, handling multiple occurrences of entities.
    
    Args:
        text: The original text to annotate
        annotation_json: JSON structure with annotations
        nlp: Optional spaCy language model (will load if not provided)
    
    Returns:
        tokens: List of tokens
        labels: List of BIO labels
        stats: Dictionary with conversion statistics
    """
    # Initialize nlp if not provided
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")
    
    doc = nlp(text)
    tokens = [token.text for token in doc]
    labels = ["O"] * len(tokens)
    
    entity_spans = []
    
    # Flatten all labeled 'value' fields from the nested JSON structure
    for category, entries in annotation_json.items():
        for entry in entries:
            for sublabel, data in entry.items():
                if isinstance(data, dict):
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
    multiple_occurrences_handled = 0
    overlapping_entities = []
    
    # Track which character positions have been labeled to detect overlaps
    char_labels = [None] * len(text)
    
    for entity in entity_spans:
        label = entity["label"]
        value = entity["value"]
        span = entity.get("span", [])
        
        matched_positions = []
        
        # First, check if we have a valid span that matches
        if isinstance(span, list) and len(span) == 2 and all(isinstance(x, int) for x in span):
            start, end = span
            # Validate span bounds
            if 0 <= start < len(text) and 0 <= end <= len(text) and text[start:end] == value:
                matched_positions = [(start, end)]
                correct_spans += 1
            else:
                # Span doesn't match, fall back to searching
                matched_positions = find_all_occurrences(text, value)
                if matched_positions:
                    fallbacks_used += 1
                    # Try to find the best match based on proximity to the original span
                    if 0 <= start < len(text):
                        # Sort by distance from the original span start
                        matched_positions.sort(key=lambda x: abs(x[0] - start))
                        # Take only the closest match if span was provided but wrong
                        matched_positions = [matched_positions[0]]
        else:
            # No valid span provided, find all occurrences
            matched_positions = find_all_occurrences(text, value)
            if matched_positions:
                fallbacks_used += 1
        
        if not matched_positions:
            not_found_entities_count += 1
            not_found_entities.append(value)
            continue
        
        if len(matched_positions) > 1:
            multiple_occurrences_handled += 1
        
        # Process each occurrence
        for start_idx, end_idx in matched_positions:
            # Check for overlapping entities
            overlap_detected = False
            for i in range(start_idx, end_idx):
                if char_labels[i] is not None:
                    overlap_detected = True
                    overlapping_entities.append({
                        "new": f"{label}:{value}",
                        "existing": char_labels[i],
                        "position": (start_idx, end_idx)
                    })
                    break
            
            # Skip if overlap detected (first entity wins)
            if overlap_detected:
                continue
            
            # Mark character positions as labeled
            for i in range(start_idx, end_idx):
                char_labels[i] = f"{label}:{value}"
            
            # Map character positions to token indices
            token_start = None
            token_end = None
            
            for i, token in enumerate(doc):
                token_start_char = token.idx
                token_end_char = token.idx + len(token.text)
                
                # Check if token overlaps with entity
                if token_start_char < end_idx and token_end_char > start_idx:
                    if token_start is None:
                        token_start = i
                    token_end = i
            
            # Apply BIO labels if we found valid token boundaries
            if token_start is not None and token_end is not None:
                for i in range(token_start, token_end + 1):
                    if i < len(labels):  # Ensure we don't go out of bounds
                        labels[i] = f"B-{label}" if i == token_start else f"I-{label}"
    
    stats = {
        "correct_spans": correct_spans,
        "fallbacks_used": fallbacks_used,
        "not_found_entities": not_found_entities,
        "not_found_entities_count": not_found_entities_count,
        "total_entities": len(entity_spans),
        "multiple_occurrences_handled": multiple_occurrences_handled,
        "overlapping_entities_count": len(overlapping_entities),
        "overlapping_entities": overlapping_entities[:5] if overlapping_entities else []  # Show first 5 overlaps
    }
    
    return tokens, labels, stats


def find_all_occurrences(text, value):
    """
    Find all occurrences of a value in text.
    
    Args:
        text: The text to search in
        value: The value to search for
    
    Returns:
        List of (start, end) tuples for each occurrence
    """
    occurrences = []
    start = 0
    
    while True:
        # Case-sensitive search by default, could be made configurable
        idx = text.find(value, start)
        if idx == -1:
            break
        occurrences.append((idx, idx + len(value)))
        start = idx + 1  # Move past this occurrence
    
    return occurrences


def find_all_occurrences_case_insensitive(text, value):
    """
    Find all occurrences of a value in text (case-insensitive).
    
    Args:
        text: The text to search in
        value: The value to search for
    
    Returns:
        List of (start, end) tuples for each occurrence
    """
    occurrences = []
    pattern = re.compile(re.escape(value), re.IGNORECASE)
    
    for match in pattern.finditer(text):
        occurrences.append((match.start(), match.end()))
    
    return occurrences


def generate_bio_from_json(text_file, annotations_file):
    with open(text_file, "r", encoding="utf-8") as f:
        text = f.read()
  
    with open(annotations_file, "r", encoding="utf-8") as f:
        annotation_json = json.load(f)
    tokens, labels, stats = convert_json_to_bio(text, annotation_json)    
    return tokens, labels, stats



if __name__ == "__main__":
    input_file_text = "/home/s27mhusa_hpc/Master-Thesis/Text_Files_For_LLM_Input/00bee634-47e6-490b-89ba-2464c9f09c31_inception.txt"
    input_file_annotations = "/home/s27mhusa_hpc/Master-Thesis/Results/Results_new_prompt_json/LLM_annotated_Qwen2.5-72B-Instruct/00bee634-47e6-490b-89ba-2464c9f09c31_inception_annotated.txt"

    generate_bio_from_json(input_file_text,input_file_annotations)