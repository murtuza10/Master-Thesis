import os
import sys
from typing import Dict, Tuple

import spacy
import torch
from cassis import Cas, load_typesystem
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
)


MODEL_PATH = \
    "/lustre/scratch/data/s27mhusa_hpc-murtuza_master_thesis/roberta-en-de_final_model_regularized_saved_specific_22"


# Keep this label list in sync with training
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


TYPE_SYSTEM_PATH = \
    "/home/s27mhusa_hpc/Master-Thesis/Evaluation_Files/full-typesystem.xml"


def build_label_mapping() -> Dict[str, Tuple[str, str]]:
    """
    Build mapping from entity labels to XMI types and feature values.
    NOTE: city, country, region map to locationName in the XMI output.
    """
    mapping: Dict[str, Tuple[str, str]] = {}

    def set_map(key: str, type_name: str, feature_value: str) -> None:
        mapping[key] = (type_name, feature_value)

    # Soil-related map to webanno.custom.Soil with Soil feature
    for k in [
        "soilTexture",
        "soilBulkDensity",
        "soilOrganicCarbon",
        "soilReferenceGroup",
        "soilAvailableNitrogen",
        "soilDepth",
        "soilPH",
    ]:
        set_map(k, "webanno.custom.Soil", k)

    # Crops map to webanno.custom.Crops with crops feature
    set_map("cropSpecies", "webanno.custom.Crops", "cropSpecies")
    set_map("cropVariety", "webanno.custom.Crops", "cropVariety")

    # Location map to webanno.custom.Location with Location feature
    # city, country, region all map to locationName
    set_map("city", "webanno.custom.Location", "locationName")
    set_map("country", "webanno.custom.Location", "locationName")
    set_map("region", "webanno.custom.Location", "locationName")
    set_map("latitude", "webanno.custom.Location", "latitude")
    set_map("longitude", "webanno.custom.Location", "longitude")

    # Time map to webanno.custom.Timestatement with Timestatement feature
    set_map("startTime", "webanno.custom.Timestatement", "startTime")
    set_map("endTime", "webanno.custom.Timestatement", "endTime")
    set_map("duration", "webanno.custom.Timestatement", "duration")

    return mapping


def normalize_group(label: str) -> str:
    """Remove B- or I- prefix from label."""
    if label.startswith("B-") or label.startswith("I-"):
        return label.split("-", 1)[1]
    return label


def load_inference_pipeline(model_path: str):
    """Load model and tokenizer with proper label configuration."""
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    
   
    id2label = {i: lab for i, lab in enumerate(LABEL_LIST)}
    label2id = {lab: i for i, lab in enumerate(LABEL_LIST)}
    
    # Check if model has generic labels and fix them
    current_labels = getattr(model.config, "id2label", {})
    has_generic_labels = (
        not current_labels or 
        all(v.startswith("LABEL_") for v in list(current_labels.values())[:5])
    )
    
    if has_generic_labels:
        print(f"⚠️  Fixing generic labels (LABEL_0, LABEL_1, ...) → actual entity labels")
        model.config.id2label = id2label
        model.config.label2id = label2id
        print(f"✅ Updated model config with {len(LABEL_LIST)} proper labels")
    else:
        print(f"✅ Model already has proper labels")
    
    # Show first few labels for verification
    print(f"First 5 model labels: {[model.config.id2label[i] for i in range(min(5, model.config.num_labels))]}")
    
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'GPU' if device == 0 else 'CPU'}\n")
    
    return pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",  # Groups B- and I- tags
        device=device,
    )


def create_cas_with_segments(text: str, ts) -> Cas:
    """Create CAS with sentence and token segmentation using spaCy."""
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    cas = Cas(typesystem=ts)
    cas.sofa_string = text

    Sentence = ts.get_type("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
    Token = ts.get_type("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token")

    for sent in doc.sents:
        cas.add(Sentence(begin=sent.start_char, end=sent.end_char))
        for token in sent:
            cas.add(Token(begin=token.idx, end=token.idx + len(token.text)))

    return cas


def add_entity_to_cas(cas: Cas, ts, start: int, end: int, type_name: str, feature_value: str) -> None:
    """Add entity annotation to CAS based on type and feature."""
    try:
        type_def = ts.get_type(type_name)
        if type_name == "webanno.custom.Soil":
            cas.add(type_def(begin=start, end=end, Soil=feature_value))
        elif type_name == "webanno.custom.Crops":
            cas.add(type_def(begin=start, end=end, crops=feature_value))
        elif type_name == "webanno.custom.Location":
            cas.add(type_def(begin=start, end=end, Location=feature_value))
        elif type_name == "webanno.custom.Timestatement":
            cas.add(type_def(begin=start, end=end, Timestatement=feature_value))
    except Exception as e:
        print(f"Warning: Could not add entity {feature_value} at [{start}:{end}]: {e}")


def process_file(txt_path: str, out_xmi_path: str, ner_pipe, label_map: Dict[str, Tuple[str, str]], ts) -> None:
    """Process single text file and generate XMI with entity annotations."""
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()

        if not text.strip():
            print(f"⚠️  Skipping empty file: {os.path.basename(txt_path)}")
            return

        # Create CAS with segmentation
        cas = create_cas_with_segments(text, ts)

        # Run NER prediction
        predictions = ner_pipe(text)
        
        entity_count = 0
        skipped_count = 0
        
        for pred in predictions:
            group = pred.get("entity_group") or pred.get("entity") or "O"
            group = normalize_group(group)
            
            if group == "O":
                continue

            # Get type mapping (city/country/region automatically map to locationName)
            type_name, feature_value = label_map.get(group, (None, None))
            if not type_name:
                skipped_count += 1
                continue

            start = int(pred["start"])
            end = int(pred["end"])
            add_entity_to_cas(cas, cas.typesystem, start, end, type_name, feature_value)
            entity_count += 1

        # Save XMI
        cas.to_xmi(out_xmi_path)
        
        status = "✅" if entity_count > 0 else "⚠️ "
        print(f"{status} {os.path.basename(txt_path)} → {entity_count} entities → {os.path.basename(out_xmi_path)}")
        
        if skipped_count > 0:
            print(f"   (skipped {skipped_count} unmapped entities)")

    except Exception as e:
        print(f"❌ Error processing {os.path.basename(txt_path)}: {e}")
        import traceback
        traceback.print_exc()


def run_inference(input_dir: str, output_dir: str, model_path: str) -> None:
    """Run inference on all text files in input directory."""
    # Validate paths
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.exists(TYPE_SYSTEM_PATH):
        print(f"Error: Type system file not found: {TYPE_SYSTEM_PATH}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.exists(model_path):
        print(f"Error: Model path not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    # Load type system
    try:
        with open(TYPE_SYSTEM_PATH, "rb") as f:
            ts = load_typesystem(f)
    except Exception as e:
        print(f"Error loading type system: {e}", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load pipeline with corrected labels
    ner_pipe = load_inference_pipeline(model_path)
    label_map = build_label_mapping()

    # Get text files
    txt_files = [name for name in os.listdir(input_dir) if name.endswith(".txt")]
    
    if not txt_files:
        print(f"Warning: No .txt files found in {input_dir}", file=sys.stderr)
        return

    print(f"{'='*60}")
    print(f"Processing {len(txt_files)} text files")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"NOTE: city/country/region → locationName in XMI output")
    print(f"{'='*60}\n")

    # Process each file
    success_count = 0
    for idx, name in enumerate(txt_files, 1):
        txt_path = os.path.join(input_dir, name)
        out_name = name.replace("_inception.txt", ".xmi")
        out_xmi_path = os.path.join(output_dir, out_name)
        
        print(f"[{idx}/{len(txt_files)}] ", end="")
        process_file(txt_path, out_xmi_path, ner_pipe, label_map, ts)
        
        # Check if file was created successfully
        if os.path.exists(out_xmi_path):
            success_count += 1

    print(f"\n{'='*60}")
    print(f"✅ Processing complete!")
    print(f"   Successfully processed: {success_count}/{len(txt_files)} files")
    print(f"   Output directory: {output_dir}")
    print(f"{'='*60}")


def main() -> None:
    """Main entry point for the inference script."""
    if len(sys.argv) < 3:
        print(
            "Usage: python infer_xlmroberta_to_xmi.py <input_txt_dir> <output_xmi_dir> [model_path]",
            file=sys.stderr,
        )
        print("\nArguments:", file=sys.stderr)
        print("  input_txt_dir   Directory containing text files to process", file=sys.stderr)
        print("  output_xmi_dir  Directory where XMI files will be saved", file=sys.stderr)
        print("  model_path      Optional: Path to trained model (default: MODEL_PATH constant)", file=sys.stderr)
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    model_path = sys.argv[3] if len(sys.argv) > 3 else MODEL_PATH

    run_inference(input_dir, output_dir, model_path)


if __name__ == "__main__":
    main()
