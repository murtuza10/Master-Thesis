import os
import sys
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional

import torch
from cassis import load_typesystem, load_cas_from_xmi
from transformers import AutoModelForTokenClassification, AutoTokenizer
from sklearn.metrics import classification_report


# Use environment variables with fallback defaults
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "/lustre/scratch/data/s27mhusa_hpc-murtuza_master_thesis/roberta-en-de_final_model_regularized_saved_specific_22"
)
TYPE_SYSTEM_PATH = os.getenv(
    "TYPE_SYSTEM_PATH",
    "/home/s27mhusa_hpc/Master-Thesis/Evaluation_Files/TypeSystem.xml"
)

# Labels matching your training data format
LABEL_LIST = [
    "O",
    "B-soilReferenceGroup",
    "I-soilReferenceGroup",
    "B-soilOrganicCarbon",
    "I-soilOrganicCarbon",
    "B-soilTexture",
    "I-soilTexture",
    "B-startTime",
    "I-startTime",
    "B-endTime",
    "I-endTime",
    "B-city",
    "I-city",
    "B-duration",
    "I-duration",
    "B-cropSpecies",
    "I-cropSpecies",
    "B-soilAvailableNitrogen",
    "I-soilAvailableNitrogen",
    "B-soilDepth",
    "I-soilDepth",
    "B-region",
    "I-region",
    "B-country",
    "I-country",
    "B-longitude",
    "I-longitude",
    "B-latitude",
    "I-latitude",
    "B-cropVariety",
    "I-cropVariety",
    "B-soilPH",
    "I-soilPH",
    "B-soilBulkDensity",
    "I-soilBulkDensity",
]


def normalize_pred_label(label: str) -> str:
    """Remove B-/I- prefix from label."""
    if label.startswith("B-") or label.startswith("I-"):
        return label.split("-", 1)[1]
    return label


def build_gold_label_mapping() -> Dict[str, str]:
    """
    Map fine-grained XMI labels to model's training labels.
    NOTE: city, country, region are mapped to locationName for evaluation.
    """
    mapping = {}
    
    # Crops
    mapping["cropSpecies"] = "cropSpecies"
    mapping["cropVariety"] = "cropVariety"
    
    # Location - map city/country/region to locationName
    mapping["locationName"] = "locationName"
    mapping["city"] = "locationName"  # Map to locationName
    mapping["country"] = "locationName"  # Map to locationName
    mapping["region"] = "locationName"  # Map to locationName
    mapping["latitude"] = "latitude"
    mapping["longitude"] = "longitude"
    
    # Time
    mapping["startTime"] = "startTime"
    mapping["endTime"] = "endTime"
    mapping["duration"] = "duration"
    
    # Soil
    mapping["soilTexture"] = "soilTexture"
    mapping["soilBulkDensity"] = "soilBulkDensity"
    mapping["soilOrganicCarbon"] = "soilOrganicCarbon"
    mapping["soilReferenceGroup"] = "soilReferenceGroup"
    mapping["soilAvailableNitrogen"] = "soilAvailableNitrogen"
    mapping["soilDepth"] = "soilDepth"
    mapping["soilPH"] = "soilPH"
    
    return mapping


def simple_whitespace_tokenize(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    """
    Simple whitespace tokenization matching training format.
    Returns tokens and their character offsets.
    """
    tokens = []
    offsets = []
    
    in_word = False
    word_start = 0
    
    for i, char in enumerate(text):
        if char.isspace():
            if in_word:
                tokens.append(text[word_start:i])
                offsets.append((word_start, i))
                in_word = False
        else:
            if not in_word:
                word_start = i
                in_word = True
    
    # Handle last word
    if in_word:
        tokens.append(text[word_start:])
        offsets.append((word_start, len(text)))
    
    return tokens, offsets


def build_model_and_tokenizer(model_path: str) -> Tuple[torch.nn.Module, AutoTokenizer, torch.device]:
    """Load model, tokenizer, and fix label mappings."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}", file=sys.stderr)
        raise
    
    # ================================
    # CRITICAL FIX: Override generic LABEL_0, LABEL_1 with actual labels
    # ================================
    id2label = {i: lab for i, lab in enumerate(LABEL_LIST)}
    label2id = {lab: i for i, lab in enumerate(LABEL_LIST)}
    
    # Check if model has generic labels and fix them
    current_labels = getattr(model.config, "id2label", {})
    has_generic_labels = all(v.startswith("LABEL_") for v in list(current_labels.values())[:5]) if current_labels else True
    
    if has_generic_labels or not current_labels:
        print(f"⚠️  Detected generic labels (LABEL_0, LABEL_1, ...)")
        print(f"⚠️  Overriding with actual entity labels")
        model.config.id2label = id2label
        model.config.label2id = label2id
        print(f"✅ Fixed: Model now has {len(LABEL_LIST)} proper labels")
    else:
        print(f"✅ Model already has proper labels")
    
    # Verify the fix worked
    print(f"\nModel configuration:")
    print(f"  Total labels: {model.config.num_labels}")
    print(f"  First 5 labels: {[model.config.id2label[i] for i in range(min(5, model.config.num_labels))]}")
    print(f"  Last 5 labels: {[model.config.id2label[i] for i in range(max(0, model.config.num_labels-5), model.config.num_labels)]}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    print(f"  Device: {device}\n")
    
    return model, tokenizer, device


def extract_gold_entities(xmi_path: str, ts, gold_label_map: Dict[str, str]) -> Tuple[str, List[Tuple[str, int, int]]]:
    """Extract gold standard entities from XMI file with label normalization."""
    try:
        with open(xmi_path, "rb") as f:
            cas = load_cas_from_xmi(f, typesystem=ts)
    except Exception as e:
        print(f"Error loading XMI file {xmi_path}: {e}", file=sys.stderr)
        raise

    text = cas.sofa_string
    if text is None:
        print(f"Warning: Empty document in {xmi_path}", file=sys.stderr)
        text = ""
    
    gold: List[Tuple[str, int, int]] = []

    def add(label: Optional[str], begin: Optional[int], end: Optional[int]) -> None:
        """Add entity to gold list with validation and normalization."""
        if label is None or begin is None or end is None:
            return
        if begin < 0 or end < 0 or begin >= end:
            return
        
        # Normalize label using mapping (city/country/region -> locationName)
        normalized_label = gold_label_map.get(label, label)
        gold.append((normalized_label, int(begin), int(end)))

    # Extract all entity types
    if ts.contains_type("webanno.custom.Crops"):
        tp = ts.get_type("webanno.custom.Crops")
        for a in cas.select(tp.name):
            crops_label = getattr(a, "crops", None)
            add(crops_label, a.begin, a.end)

    if ts.contains_type("webanno.custom.Soil"):
        tp = ts.get_type("webanno.custom.Soil")
        for a in cas.select(tp.name):
            soil_label = getattr(a, "Soil", None)
            add(soil_label, a.begin, a.end)

    if ts.contains_type("webanno.custom.Location"):
        tp = ts.get_type("webanno.custom.Location")
        for a in cas.select(tp.name):
            location_label = getattr(a, "Location", None)
            add(location_label, a.begin, a.end)

    if ts.contains_type("webanno.custom.Timestatement"):
        tp = ts.get_type("webanno.custom.Timestatement")
        for a in cas.select(tp.name):
            time_label = getattr(a, "Timestatement", None)
            add(time_label, a.begin, a.end)

    return text, gold


def extract_pred_entities(text: str, model, tokenizer, device) -> List[Tuple[str, int, int]]:
    """
    Extract predicted entities using word-level tokenization matching training format.
    Maps city, country, region to locationName AFTER prediction.
    """
    if not text:
        return []
    
    # Tokenize into words first (matching training format)
    words, word_offsets = simple_whitespace_tokenize(text)
    
    if not words:
        return []
    
    # Tokenize with subwords, keeping track of word boundaries
    encoding = tokenizer(
        words,
        is_split_into_words=True,  # CRITICAL: Match training format
        truncation=True,
        max_length=256,  # Match training max_length
        padding=True,
        return_tensors="pt",
    )
    
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    # Get predictions
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        predictions = torch.argmax(logits, dim=-1).cpu().tolist()[0]
    
    # Map predictions back to words using fixed id2label
    word_ids = encoding.word_ids(batch_index=0)
    id2label = model.config.id2label  # Now contains actual labels, not LABEL_0
    
    # Collect entities at word level
    current_entity = None
    entities = []
    previous_word_idx = None
    
    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None:  # Special tokens
            continue
        
        # Only consider first subword token for each word
        if word_idx == previous_word_idx:
            continue
        previous_word_idx = word_idx
        
        label = id2label.get(predictions[token_idx], "O")
        
        if label == "O":
            if current_entity is not None:
                entities.append(current_entity)
                current_entity = None
        
        elif label.startswith("B-"):
            if current_entity is not None:
                entities.append(current_entity)
            
            entity_type = normalize_pred_label(label)
            
            char_start, char_end = word_offsets[word_idx]
            current_entity = {
                'type': entity_type,
                'start': char_start,
                'end': char_end,
                'word_start': word_idx,
                'word_end': word_idx
            }
        
        elif label.startswith("I-"):
            if current_entity is not None:
                entity_type = normalize_pred_label(label)
                
                if entity_type == current_entity['type']:
                    # Extend entity span
                    current_entity['word_end'] = word_idx
                    _, char_end = word_offsets[word_idx]
                    current_entity['end'] = char_end
                else:
                    # Different type, start new entity
                    entities.append(current_entity)
                    char_start, char_end = word_offsets[word_idx]
                    current_entity = {
                        'type': entity_type,
                        'start': char_start,
                        'end': char_end,
                        'word_start': word_idx,
                        'word_end': word_idx
                    }
            else:
                # I- without B-, treat as start of entity
                entity_type = normalize_pred_label(label)
                char_start, char_end = word_offsets[word_idx]
                current_entity = {
                    'type': entity_type,
                    'start': char_start,
                    'end': char_end,
                    'word_start': word_idx,
                    'word_end': word_idx
                }
    
    # Don't forget last entity
    if current_entity is not None:
        entities.append(current_entity)
    
    # Convert to tuple format and MAP city/country/region to locationName
    result = []
    for ent in entities:
        entity_type = ent['type']
        
        # Map city, country, region to locationName for evaluation
        if entity_type in {"city", "country", "region"}:
            entity_type = "locationName"
        
        if entity_type != "O":
            result.append((entity_type, ent['start'], ent['end']))
    
    return result


def evaluate(gold_xmi_dir: str, model_path: str, show: bool = False) -> None:
    """Evaluate model predictions against gold standard XMI files using sklearn."""
    # Validate paths
    if not os.path.exists(gold_xmi_dir):
        print(f"Error: Gold XMI directory not found: {gold_xmi_dir}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.exists(TYPE_SYSTEM_PATH):
        print(f"Error: Type system file not found: {TYPE_SYSTEM_PATH}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.exists(model_path):
        print(f"Error: Model path not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(TYPE_SYSTEM_PATH, "rb") as f:
            ts = load_typesystem(f)
    except Exception as e:
        print(f"Error loading type system: {e}", file=sys.stderr)
        sys.exit(1)

    # Build label mappings
    gold_label_map = build_gold_label_mapping()
    
    # Load model with fixed labels
    model, tokenizer, device = build_model_and_tokenizer(model_path)

    # Track entity-level matches for exact span evaluation
    agg_tp = defaultdict(int)
    agg_fp = defaultdict(int)
    agg_fn = defaultdict(int)
    
    # Track label counts
    gold_label_counts = defaultdict(int)
    pred_label_counts = defaultdict(int)
    
    # For token-level classification
    all_token_gold_labels = []
    all_token_pred_labels = []

    files = [p for p in os.listdir(gold_xmi_dir) if p.endswith(".xmi")]
    
    if not files:
        print(f"Warning: No XMI files found in {gold_xmi_dir}", file=sys.stderr)
        return
    
    print(f"{'='*60}")
    print(f"Processing {len(files)} XMI files...")
    print(f"NOTE: Mapping city, country, region → locationName for evaluation")
    print(f"{'='*60}\n")
    
    for idx, fname in enumerate(files, 1):
        xmi_path = os.path.join(gold_xmi_dir, fname)
        
        try:
            text, gold = extract_gold_entities(xmi_path, ts, gold_label_map)
            pred = extract_pred_entities(text, model, tokenizer, device)
        except Exception as e:
            print(f"Error processing {fname}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            continue

        gold_labels = sorted({lab for lab, _, _ in gold})
        pred_labels = sorted({lab for lab, _, _ in pred})
        
        print(f"[{idx}/{len(files)}] {fname}")
        print(f"  Gold: {len(gold)} entities - {gold_labels}")
        print(f"  Pred: {len(pred)} entities - {pred_labels}")

        if show:
            print(f"\n  === Gold Entities ===")
            for lab, b, e in sorted(gold, key=lambda x: (x[1], x[2], x[0])):
                snippet = text[b:e].replace("\n", " ")[:50]
                print(f"    {lab:22s} [{b:4d}:{e:4d}] \"{snippet}\"")
            
            print(f"\n  === Predicted Entities ===")
            for lab, b, e in sorted(pred, key=lambda x: (x[1], x[2], x[0])):
                snippet = text[b:e].replace("\n", " ")[:50]
                print(f"    {lab:22s} [{b:4d}:{e:4d}] \"{snippet}\"")
            print()

        # Count gold and predicted labels
        for label, _, _ in gold:
            gold_label_counts[label] += 1
        
        for label, _, _ in pred:
            pred_label_counts[label] += 1
        
        # ========================================
        # TOKEN-LEVEL ALIGNMENT FOR SKLEARN
        # ========================================
        # Tokenize text into words
        words, word_offsets = simple_whitespace_tokenize(text)
        
        # Create label array for each word position
        word_gold_labels = ['O'] * len(words)
        word_pred_labels = ['O'] * len(words)
        
        # Assign gold labels to words
        for label, start, end in gold:
            for word_idx, (w_start, w_end) in enumerate(word_offsets):
                # Check if word overlaps with entity span
                if not (w_end <= start or w_start >= end):
                    word_gold_labels[word_idx] = label
        
        # Assign predicted labels to words
        for label, start, end in pred:
            for word_idx, (w_start, w_end) in enumerate(word_offsets):
                # Check if word overlaps with entity span
                if not (w_end <= start or w_start >= end):
                    word_pred_labels[word_idx] = label
        
        # Add to overall token-level lists
        all_token_gold_labels.extend(word_gold_labels)
        all_token_pred_labels.extend(word_pred_labels)
        
        # ========================================
        # ENTITY-LEVEL EXACT SPAN MATCHING
        # ========================================
        gold_set = set(gold)
        pred_set = set(pred)

        for item in pred_set & gold_set:
            if item[0] is not None:
                agg_tp[item[0]] += 1
        for item in pred_set - gold_set:
            if item[0] is not None:
                agg_fp[item[0]] += 1
        for item in gold_set - pred_set:
            if item[0] is not None:
                agg_fn[item[0]] += 1

    # ========================================
    # GOLD LABEL COUNTS
    # ========================================
    print("\n" + "="*80)
    print("GOLD STANDARD LABEL DISTRIBUTION")
    print("="*80)
    total_gold = sum(gold_label_counts.values())
    print(f"Total Gold Entities: {total_gold}\n")
    
    for label in sorted(gold_label_counts.keys()):
        count = gold_label_counts[label]
        percentage = (count / total_gold * 100) if total_gold > 0 else 0
        print(f"{label:30s}: {count:6d} ({percentage:5.2f}%)")

    # ========================================
    # SKLEARN CLASSIFICATION REPORT (TOKEN-LEVEL)
    # ========================================
    
    print("\n" + "="*80)
    print("SKLEARN CLASSIFICATION REPORT (Token-Level)")
    print("="*80)
    print(f"NOTE: This evaluates at word/token level, treating each word as a sample")
    print(f"      Words are labeled with entity type if they overlap with an entity span\n")
    
    # Get unique labels
    unique_labels = sorted(set(all_token_gold_labels + all_token_pred_labels) - {'O'})
    
    if all_token_gold_labels and all_token_pred_labels and len(all_token_gold_labels) == len(all_token_pred_labels):
        # Print classification report
        report = classification_report(
            all_token_gold_labels, 
            all_token_pred_labels, 
            labels=unique_labels,
            zero_division=0,
            digits=4
        )
        print(report)
        
        # Also print as dictionary for programmatic access
        report_dict = classification_report(
            all_token_gold_labels,
            all_token_pred_labels,
            labels=unique_labels,
            output_dict=True,
            zero_division=0
        )
        
        print("\n" + "="*80)
        print("SKLEARN METRICS SUMMARY (Token-Level)")
        print("="*80)
        print(f"Total Tokens Evaluated: {len(all_token_gold_labels)}")
        print(f"Gold Entity Tokens:     {sum(1 for label in all_token_gold_labels if label != 'O')}")
        print(f"Predicted Entity Tokens: {sum(1 for label in all_token_pred_labels if label != 'O')}")
        print(f"\nMacro Average:")
        print(f"  Precision: {report_dict['macro avg']['precision']:.4f}")
        print(f"  Recall:    {report_dict['macro avg']['recall']:.4f}")
        print(f"  F1-Score:  {report_dict['macro avg']['f1-score']:.4f}")
        print(f"\nWeighted Average:")
        print(f"  Precision: {report_dict['weighted avg']['precision']:.4f}")
        print(f"  Recall:    {report_dict['weighted avg']['recall']:.4f}")
        print(f"  F1-Score:  {report_dict['weighted avg']['f1-score']:.4f}")
    else:
        print("Could not generate sklearn report - misaligned data")

    # ========================================
    # ENTITY-LEVEL EXACT SPAN MATCHING
    # ========================================
    
    def prf1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return prec, rec, f1

    micro_tp = sum(agg_tp.values())
    micro_fp = sum(agg_fp.values())
    micro_fn = sum(agg_fn.values())
    micro_p, micro_r, micro_f1 = prf1(micro_tp, micro_fp, micro_fn)

    print("\n" + "="*80)
    print("ENTITY-LEVEL EXACT SPAN MATCHING (Type + Boundaries)")
    print("="*80)
    print(f"NOTE: This requires exact match of both entity type AND span boundaries\n")
    print(f"Total True Positives:  {micro_tp}")
    print(f"Total False Positives: {micro_fp}")
    print(f"Total False Negatives: {micro_fn}")
    print(f"\nMicro Precision: {micro_p:.4f}")
    print(f"Micro Recall:    {micro_r:.4f}")
    print(f"Micro F1:        {micro_f1:.4f}")

    # Per-label metrics for exact span matching
    all_entity_labels = sorted(
        {lab for lab in list(agg_tp.keys()) + list(agg_fp.keys()) + list(agg_fn.keys()) if lab is not None}
    )
    
    if all_entity_labels:
        print("\n" + "="*80)
        print("PER-LABEL METRICS (Exact Span Match)")
        print("="*80)
        print(f"{'Label':<26} {'Precision':<12} {'Recall':<12} {'F1':<12} {'TP/FP/FN'}")
        print("-"*80)
        
        for lab in all_entity_labels:
            p, r, f1 = prf1(agg_tp[lab], agg_fp[lab], agg_fn[lab])
            print(f"{lab:<26} {p:<12.4f} {r:<12.4f} {f1:<12.4f} {agg_tp[lab]}/{agg_fp[lab]}/{agg_fn[lab]}")
    
    print("="*80)


def main() -> None:
    """Main entry point."""
    if len(sys.argv) < 2:
        print(
            "Usage: python eval_xlmroberta_against_xmi.py <gold_xmi_dir> [model_path] [--show]",
            file=sys.stderr,
        )
        print("\nOptions:", file=sys.stderr)
        print("  gold_xmi_dir    Directory containing gold standard XMI files", file=sys.stderr)
        print("  model_path      Path to trained model (optional)", file=sys.stderr)
        print("  --show          Show detailed entity listings", file=sys.stderr)
        sys.exit(1)

    args = [a for a in sys.argv[1:] if a != "--show"]
    show = "--show" in sys.argv[1:]

    gold_dir = args[0]
    model_path = args[1] if len(args) > 1 else MODEL_PATH
    
    evaluate(gold_dir, model_path, show=show)


if __name__ == "__main__":
    main()
