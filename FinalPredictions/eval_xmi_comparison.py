import os
import sys
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional

import torch
from cassis import load_typesystem, load_cas_from_xmi
from sklearn.metrics import classification_report


TYPE_SYSTEM_PATH = os.getenv(
    "TYPE_SYSTEM_PATH",
    "/home/s27mhusa_hpc/Master-Thesis/Evaluation_Files/TypeSystem.xml"
)


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
    mapping["city"] = "locationName"
    mapping["country"] = "locationName"
    mapping["region"] = "locationName"
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


def extract_entities_from_xmi(xmi_path: str, ts, label_map: Dict[str, str]) -> Tuple[str, List[Tuple[str, int, int]]]:
    """Extract entities from XMI file with label normalization."""
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
    
    entities: List[Tuple[str, int, int]] = []

    def add(label: Optional[str], begin: Optional[int], end: Optional[int]) -> None:
        """Add entity to list with validation and normalization."""
        if label is None or begin is None or end is None:
            return
        if begin < 0 or end < 0 or begin >= end:
            return
        
        # Normalize label using mapping (city/country/region -> locationName)
        normalized_label = label_map.get(label, label)
        entities.append((normalized_label, int(begin), int(end)))

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

    return text, entities


def find_matching_files(gold_dir: str, pred_dir: str) -> List[Tuple[str, str]]:
    """
    Find matching XMI files between gold and predicted directories.
    Returns list of (gold_path, pred_path) tuples.
    """
    gold_files = {f: os.path.join(gold_dir, f) for f in os.listdir(gold_dir) if f.endswith(".xmi")}
    pred_files = {f: os.path.join(pred_dir, f) for f in os.listdir(pred_dir) if f.endswith(".xmi")}
    
    # Find common files
    common_files = set(gold_files.keys()) & set(pred_files.keys())
    
    if not common_files:
        # Try matching with different naming patterns (e.g., with/without _inception suffix)
        gold_base_names = {f.replace("_inception.xmi", ".xmi").replace(".xmi", ""): f for f in gold_files.keys()}
        pred_base_names = {f.replace("_inception.xmi", ".xmi").replace(".xmi", ""): f for f in pred_files.keys()}
        
        common_base = set(gold_base_names.keys()) & set(pred_base_names.keys())
        
        matching_pairs = [
            (gold_files[gold_base_names[base]], pred_files[pred_base_names[base]])
            for base in common_base
        ]
    else:
        matching_pairs = [(gold_files[f], pred_files[f]) for f in common_files]
    
    return matching_pairs


def evaluate(gold_xmi_dir: str, pred_xmi_dir: str, show: bool = False) -> None:
    """Evaluate predicted XMI files against gold standard XMI files."""
    # Validate paths
    if not os.path.exists(gold_xmi_dir):
        print(f"Error: Gold XMI directory not found: {gold_xmi_dir}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.exists(pred_xmi_dir):
        print(f"Error: Predicted XMI directory not found: {pred_xmi_dir}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.exists(TYPE_SYSTEM_PATH):
        print(f"Error: Type system file not found: {TYPE_SYSTEM_PATH}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(TYPE_SYSTEM_PATH, "rb") as f:
            ts = load_typesystem(f)
    except Exception as e:
        print(f"Error loading type system: {e}", file=sys.stderr)
        sys.exit(1)

    # Build label mappings
    label_map = build_gold_label_mapping()
    
    # Find matching files
    matching_pairs = find_matching_files(gold_xmi_dir, pred_xmi_dir)
    
    if not matching_pairs:
        print(f"Error: No matching XMI files found between directories", file=sys.stderr)
        print(f"  Gold dir: {gold_xmi_dir}", file=sys.stderr)
        print(f"  Pred dir: {pred_xmi_dir}", file=sys.stderr)
        sys.exit(1)

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

    print(f"{'='*60}")
    print(f"Processing {len(matching_pairs)} matching XMI file pairs...")
    print(f"Gold directory: {gold_xmi_dir}")
    print(f"Pred directory: {pred_xmi_dir}")
    print(f"NOTE: Mapping city, country, region → locationName for evaluation")
    print(f"{'='*60}\n")
    
    for idx, (gold_path, pred_path) in enumerate(matching_pairs, 1):
        gold_fname = os.path.basename(gold_path)
        pred_fname = os.path.basename(pred_path)
        
        try:
            text_gold, gold = extract_entities_from_xmi(gold_path, ts, label_map)
            text_pred, pred = extract_entities_from_xmi(pred_path, ts, label_map)
            
            # Verify texts match
            if text_gold != text_pred:
                print(f"Warning: Text mismatch in {gold_fname} vs {pred_fname}", file=sys.stderr)
                # Use gold text as reference
                text = text_gold
            else:
                text = text_gold
                
        except Exception as e:
            print(f"Error processing {gold_fname}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            continue

        gold_labels = sorted({lab for lab, _, _ in gold})
        pred_labels = sorted({lab for lab, _, _ in pred})
        
        print(f"[{idx}/{len(matching_pairs)}] {gold_fname}")
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
    # PREDICTED LABEL COUNTS
    # ========================================
    print("\n" + "="*80)
    print("PREDICTED LABEL DISTRIBUTION")
    print("="*80)
    total_pred = sum(pred_label_counts.values())
    print(f"Total Predicted Entities: {total_pred}\n")
    
    for label in sorted(pred_label_counts.keys()):
        count = pred_label_counts[label]
        percentage = (count / total_pred * 100) if total_pred > 0 else 0
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
    if len(sys.argv) < 3:
        print(
            "Usage: python eval_xmi_comparison.py <gold_xmi_dir> <pred_xmi_dir> [--show]",
            file=sys.stderr,
        )
        print("\nArguments:", file=sys.stderr)
        print("  gold_xmi_dir    Directory containing gold standard XMI files", file=sys.stderr)
        print("  pred_xmi_dir    Directory containing predicted XMI files", file=sys.stderr)
        print("\nOptions:", file=sys.stderr)
        print("  --show          Show detailed entity listings for each file", file=sys.stderr)
        sys.exit(1)

    args = [a for a in sys.argv[1:] if a != "--show"]
    show = "--show" in sys.argv[1:]

    if len(args) < 2:
        print("Error: Both gold and predicted directories are required", file=sys.stderr)
        sys.exit(1)

    gold_dir = args[0]
    pred_dir = args[1]
    
    evaluate(gold_dir, pred_dir, show=show)


if __name__ == "__main__":
    main()
