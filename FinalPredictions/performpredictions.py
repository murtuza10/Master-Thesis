import os
import re
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from cassis import Cas, load_typesystem


# ==============================
# CONFIGURATION
# ==============================

MODEL_PATH = "/lustre/scratch/data/s27mhusa_hpc-murtuza_master_thesis/roberta-en-de_final_model_regularized_saved_specific_22"
INPUT_DIR = "/home/s27mhusa_hpc/Master-Thesis/Text_Files_Test_Data"
OUTPUT_DIR = "/home/s27mhusa_hpc/Master-Thesis/Test_Model_Predictions_XMI"
TYPESYSTEM_PATH = "/home/s27mhusa_hpc/Master-Thesis/Evaluation_Files/full-typesystem.xml"

# Your actual label list in the exact order used during training
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


# ==============================
# MODEL INITIALIZATION WITH FIXED LABELS
# ==============================

def load_model_for_prediction(model_path, label_list):
    """Load model and fix the label mapping"""
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_path)

    # Fix the id2label and label2id mappings
    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in enumerate(label_list)}
    
    # Update model config with correct labels
    model.config.id2label = id2label
    model.config.label2id = label2id
    
    print(f"✅ Model loaded with {len(label_list)} labels")
    print(f"First 5 labels: {label_list[:5]}")
    print(f"Last 5 labels: {label_list[-5:]}")
    
    # Create NER pipeline
    ner_pipeline = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=0 if torch.cuda.is_available() else -1
    )
    
    print(f"Using device: {'GPU' if torch.cuda.is_available() else 'CPU'}\n")
    return ner_pipeline, tokenizer


# ==============================
# LABEL MAPPING FUNCTION
# ==============================

def map_entity_label(entity_group):
    """Map entity groups to XMI annotation types"""
    if entity_group in ["cropSpecies", "cropVariety"]:
        return ("webanno.custom.Crops", entity_group)
    elif entity_group in ["city", "country", "region", "latitude", "longitude"]:
        location_subtype = "locationName" if entity_group in ["city", "country", "region"] else entity_group
        return ("webanno.custom.Location", location_subtype)
    elif entity_group in ["startTime", "endTime", "duration"]:
        return ("webanno.custom.Timestatement", entity_group)
    elif entity_group in [
        "soilReferenceGroup", "soilOrganicCarbon", "soilTexture",
        "soilAvailableNitrogen", "soilDepth", "soilPH", "soilBulkDensity"
    ]:
        return ("webanno.custom.Soil", entity_group)
    return (None, None)


# ==============================
# PREDICTION & XMI ANNOTATION
# ==============================

def predict_and_annotate_text(input_file, output_file, ner_pipeline):
    """Run NER prediction and create XMI annotations"""
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # Clean text
        text = re.sub(r'\s+', ' ', raw_text.strip())
        if not text:
            print(f"⚠️  Empty file: {os.path.basename(input_file)}")
            return False

        print(f"Processing: {os.path.basename(input_file)}")

        # Split into sentences for better context
        sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+', text) if s.strip()]
        
        # Run NER on each sentence
        all_predictions = []
        for sent in sentences:
            preds = ner_pipeline(sent)
            # Adjust offsets to match full text
            sent_start = text.find(sent)
            for pred in preds:
                pred['start'] += sent_start
                pred['end'] += sent_start
                all_predictions.append(pred)

        if not all_predictions:
            print(f"⚠️  No entities found: {os.path.basename(input_file)}")
            return False

        # Load typesystem
        with open(TYPESYSTEM_PATH, "rb") as f:
            ts = load_typesystem(f)

        # Create CAS
        cas = Cas(typesystem=ts)
        cas.sofa_string = text

        # Get type definitions
        Sentence = ts.get_type("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
        Token = ts.get_type("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token")
        CropsEntity = ts.get_type("webanno.custom.Crops")
        LocationEntity = ts.get_type("webanno.custom.Location")
        TimeEntity = ts.get_type("webanno.custom.Timestatement")
        SoilEntity = ts.get_type("webanno.custom.Soil")

        # Add sentence and token annotations
        offset = 0
        for sent_text in sentences:
            start = text.find(sent_text, offset)
            end = start + len(sent_text)
            cas.add(Sentence(begin=start, end=end))
            
            # Tokenize sentence
            token_offset = start
            for token_text in sent_text.split():
                t_start = text.find(token_text, token_offset)
                t_end = t_start + len(token_text)
                cas.add(Token(begin=t_start, end=t_end))
                token_offset = t_end
            
            offset = end

        # Add entity annotations
        annotated_spans = []
        entity_count = 0

        for entity in all_predictions:
            start = entity['start']
            end = entity['end']
            entity_group = entity['entity_group']
            
            # Skip overlapping entities
            if any(s <= start < e or s < end <= e for s, e in annotated_spans):
                continue
            
            # Map to XMI type
            ann_type, subtype = map_entity_label(entity_group)
            if ann_type is None:
                continue

            # Create annotation
            if ann_type == "webanno.custom.Crops":
                cas.add(CropsEntity(begin=start, end=end, crops=subtype))
            elif ann_type == "webanno.custom.Location":
                cas.add(LocationEntity(begin=start, end=end, Location=subtype))
            elif ann_type == "webanno.custom.Timestatement":
                cas.add(TimeEntity(begin=start, end=end, Timestatement=subtype))
            elif ann_type == "webanno.custom.Soil":
                cas.add(SoilEntity(begin=start, end=end, Soil=subtype))

            annotated_spans.append((start, end))
            entity_count += 1

        if entity_count == 0:
            print(f"⚠️  No valid entities after mapping: {os.path.basename(input_file)}")
            return False

        # Save XMI
        cas.to_xmi(output_file)
        print(f"✅ Found {entity_count} entities → {os.path.basename(output_file)}")
        return True

    except Exception as e:
        print(f"❌ Error processing {os.path.basename(input_file)}: {e}")
        import traceback
        traceback.print_exc()
        return False


# ==============================
# BATCH PROCESSING
# ==============================

def process_directory(input_dir, output_dir, ner_pipeline):
    """Process all text files in directory"""
    os.makedirs(output_dir, exist_ok=True)
    
    txt_files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
    if not txt_files:
        print(f"⚠️  No .txt files found in {input_dir}")
        return

    print(f"{'='*60}")
    print(f"Processing {len(txt_files)} files")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    success_count = 0
    failed_count = 0

    for idx, filename in enumerate(txt_files, 1):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace(".txt", "_model.xmi"))
        
        print(f"[{idx}/{len(txt_files)}] ", end="")
        
        if predict_and_annotate_text(input_path, output_path, ner_pipeline):
            success_count += 1
        else:
            failed_count += 1

    # Summary
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Success: {success_count}/{len(txt_files)}")
    print(f"❌ Failed: {failed_count}/{len(txt_files)}")
    print(f"📁 Output directory: {output_dir}")
    print(f"{'='*60}\n")


# ==============================
# MAIN EXECUTION
# ==============================

def main():
    print(f"\n{'='*60}")
    print("NER MODEL → XMI CONVERTER")
    print(f"{'='*60}\n")
    
    # Load model with corrected labels
    ner_pipeline, tokenizer = load_model_for_prediction(MODEL_PATH, LABEL_LIST)
    
    # Process all files
    process_directory(INPUT_DIR, OUTPUT_DIR, ner_pipeline)
    
    print("✅ All processing completed!\n")


if __name__ == "__main__":
    main()
