import spacy
from typing import Tuple, List, Dict, Any

# Load spaCy's small English model
nlp = spacy.load("en_core_web_sm")

# Define the exact label list as provided
LABEL_LIST = [
    "O",
    "B-soilReferenceGroup", "I-soilReferenceGroup",
    "B-soilOrganicCarbon", "I-soilOrganicCarbon",
    "B-soilTexture", "I-soilTexture",
    "B-startTime", "I-startTime",
    "B-endTime", "I-endTime",
    "B-city", "I-city",
    "B-duration", "I-duration",
    "B-cropSpecies", "I-cropSpecies",
    "B-soilAvailableNitrogen", "I-soilAvailableNitrogen",
    "B-soilDepth", "I-soilDepth",
    "B-region", "I-region",
    "B-country", "I-country",
    "B-longitude", "I-longitude",
    "B-latitude", "I-latitude",
    "B-cropVariety", "I-cropVariety",
    "B-soilPH", "I-soilPH",
    "B-soilBulkDensity", "I-soilBulkDensity"
]

def convert_json_to_bio(text: str, annotation_json: Dict[str, Any]) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """
    Convert nested JSON annotations to BIO format while maintaining original structure.
    
    Args:
        text: The input text to be processed
        annotation_json: Dictionary containing annotations in the nested format:
            {
                "id": 12,
                "gold_output": {
                    "Crops": [],
                    "Soil": [
                        {"soilDepth": {"value": "depth", "span": [71, 76]}},
                        {"soilDepth": {"value": "100 cm", "span": [80, 86]}},
                        ...
                    ],
                    "Location": [],
                    "Time Statement": []
                }
            }
    
    Returns:
        Tuple containing:
        - List of tokens
        - List of BIO tags (using exact label list)
        - Statistics dictionary
    """
    doc = nlp(text)
    tokens = [token.text for token in doc]
    labels = ["O"] * len(tokens)

    # Statistics tracking
    stats = {
        "correct_spans": 0,
        "fallbacks_used": 0,
        "not_found_entities": [],
        "not_found_entities_count": 0,
        "total_entities": 0,
        "invalid_labels": []
    }

    # Create mapping from sublabel to valid BIO tags
    valid_labels = set(LABEL_LIST)
    sublabel_to_tag = {
        "soilReferenceGroup": "soilReferenceGroup",
        "soilOrganicCarbon": "soilOrganicCarbon",
        "soilTexture": "soilTexture",
        "startTime": "startTime",
        "endTime": "endTime",
        "city": "city",
        "duration": "duration",
        "cropSpecies": "cropSpecies",
        "soilAvailableNitrogen": "soilAvailableNitrogen",
        "soilDepth": "soilDepth",
        "region": "region",
        "country": "country",
        "longitude": "longitude",
        "latitude": "latitude",
        "cropVariety": "cropVariety",
        "soilPH": "soilPH",
        "soilBulkDensity": "soilBulkDensity"
    }

    # Extract the actual annotations
    if "gold_output" in annotation_json:
        annotations = annotation_json["gold_output"]
    else:
        annotations = annotation_json

    # Process each category
    for category, entity_list in annotations.items():
        if not isinstance(entity_list, list):
            continue
            
        for entity_dict in entity_list:
            if not isinstance(entity_dict, dict):
                continue
                
            for sublabel, value_info in entity_dict.items():
                # Check if sublabel is in our mapping
                if sublabel not in sublabel_to_tag:
                    stats["invalid_labels"].append(sublabel)
                    continue
                    
                tag_base = sublabel_to_tag[sublabel]
                
                if not isinstance(value_info, dict):
                    continue
                    
                value = value_info.get("value", "")
                span = value_info.get("span", [])
                
                if not value:
                    continue
                    
                stats["total_entities"] += 1
                
                # Try to find the value in text
                if isinstance(span, list) and len(span) == 2:
                    start_idx, end_idx = span
                    if text[start_idx:end_idx] == value:
                        stats["correct_spans"] += 1
                    else:
                        start_idx = text.find(value)
                        end_idx = start_idx + len(value) if start_idx != -1 else -1
                else:
                    start_idx = text.find(value)
                    end_idx = start_idx + len(value) if start_idx != -1 else -1
                
                if start_idx == -1 or end_idx == -1:
                    stats["not_found_entities_count"] += 1
                    stats["not_found_entities"].append(f"{tag_base}: {value}")
                    continue
                
                # Map character positions to token indices
                token_start = token_end = -1
                for i, token in enumerate(doc):
                    if token_start == -1 and token.idx >= start_idx:
                        token_start = i
                    if token.idx + len(token.text) >= end_idx:
                        token_end = i
                        break

                if token_start != -1 and token_end != -1:
                    # Mark tokens with BIO tags
                    for i in range(token_start, token_end + 1):
                        prefix = "B-" if i == token_start else "I-"
                        bio_tag = prefix + tag_base
                        
                        # Verify the tag is in our allowed list
                        if bio_tag in valid_labels:
                            labels[i] = bio_tag
                        else:
                            stats["invalid_labels"].append(bio_tag)
                    
                    if not (isinstance(span, list) and len(span) == 2 and text[span[0]:span[1]] == value):
                        stats["fallbacks_used"] += 1

    return tokens, labels, stats

def generate_bio_from_json(text: str, annotation_json: Dict[str, Any]) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """
    Wrapper function to convert nested JSON annotations to BIO format using exact label list.
    
    Args:
        text: Input text string
        annotation_json: Dictionary containing the annotations in nested format
    
    Returns:
        Tuple of (tokens, BIO tags, statistics)
    """
    return convert_json_to_bio(text, annotation_json)