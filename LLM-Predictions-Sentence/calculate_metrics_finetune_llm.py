import os
import json
import evaluate
from seqeval.metrics import classification_report
from generate_bio_from_json_finetune import generate_bio_from_json
from typing import Dict, Any

import json
from datetime import datetime

def save_comparison_files(pred_ann: Dict[str, Any], gold_ann: Dict[str, Any], 
                         text: str, output_dir: str, item_id: str = None) -> None:
    """
    Save prediction and gold annotations to files for comparison.
    
    Args:
        pred_ann: Normalized prediction annotations
        gold_ann: Gold standard annotations
        text: The original text
        output_dir: Directory to save the files
        item_id: Optional ID for the item (used in filenames)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"comparison_{item_id}_{timestamp}" if item_id else f"comparison_{timestamp}"
    
    # Create comparison data structure
    comparison_data = {
        "text": text,
        "timestamp": timestamp,
        "predictions": pred_ann,
        "gold_standard": gold_ann
    }
    
    # Save to JSON file
    json_path = os.path.join(output_dir, f"{base_filename}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)
    
    # Save to human-readable text file
    txt_path = os.path.join(output_dir, f"{base_filename}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"=== Comparison for Item {item_id or 'N/A'} ===\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write("=== Original Text ===\n")
        f.write(f"{text}\n\n")
        
        f.write("=== Predictions ===\n")
        f.write(json.dumps(pred_ann, indent=2, ensure_ascii=False) + "\n\n")
        
        f.write("=== Gold Standard ===\n")
        f.write(json.dumps(gold_ann, indent=2, ensure_ascii=False) + "\n\n")
        
        # Add differences section
        f.write("=== Differences ===\n")
        for category in ["Crops", "Soil", "Location", "Time Statement"]:
            pred_items = pred_ann.get(category, [])
            gold_items = gold_ann.get(category, [])
            
            if pred_items != gold_items:
                f.write(f"Category: {category}\n")
                f.write(f"Predictions: {json.dumps(pred_items, ensure_ascii=False)}\n")
                f.write(f"Gold: {json.dumps(gold_items, ensure_ascii=False)}\n\n")

    print(f"Saved comparison files to:\n- {json_path}\n- {txt_path}")

def normalize_pred_entities(pred_entry: Dict) -> Dict:
    """
    Normalize prediction entities to match the gold standard format while preserving structure.
    
    Args:
        pred_entry: Prediction entry in format:
            {
                "id": 1,
                "entities": {
                    "crops": [
                        {"cropSpecies": {"value": "wheat", "span": [11, 16]}},
                        ...
                    ],
                    "soil": [
                        {"soilPH": {"value": "6.5", "span": [30, 33]}},
                        ...
                    ],
                    ...
                }
            }
    
    Returns:
        Normalized dictionary in format:
            {
                "Crops": [
                    {"cropSpecies": {"value": "wheat", "span": [11, 16]}},
                    ...
                ],
                "Soil": [
                    {"soilPH": {"value": "6.5", "span": [30, 33]}},
                    ...
                ],
                "Location": [],
                "Time Statement": []
            }
    """
    if not isinstance(pred_entry, dict):
        return {
            "Crops": [],
            "Soil": [],
            "Location": [],
            "Time Statement": []
        }

    raw_entities = pred_entry.get("entities", {})
    result = {
        "Crops": [],
        "Soil": [],
        "Location": [],
        "Time Statement": []
    }

    # Mapping from prediction keys to standardized categories
    category_map = {
        "crops": "Crops",
        "soil": "Soil",
        "location": "Location",
        "timestatement": "Time Statement"
    }

    # Valid sublabels for each category that match our label list
    valid_sublabels = {
        "Crops": ["cropSpecies", "cropVariety"],
        "Soil": [
            "soilReferenceGroup", "soilOrganicCarbon", "soilTexture",
            "soilPH", "soilBulkDensity", "soilAvailableNitrogen",
            "soilDepth"
        ],
        "Location": ["city", "region", "country", "longitude", "latitude"],
        "Time Statement": ["startTime", "endTime", "duration"]
    }

    for raw_cat, entries in raw_entities.items():
        # Skip if category not in our mapping
        normalized_cat = category_map.get(raw_cat.lower())
        if not normalized_cat:
            continue

        # Skip if entries is not a list
        if not isinstance(entries, list):
            continue

        for entry in entries:
            if not isinstance(entry, dict):
                continue

            normalized_entry = {}
            for sublabel, value_info in entry.items():
                # Skip if sublabel not valid for this category
                if sublabel not in valid_sublabels.get(normalized_cat, []):
                    continue

                # Handle both formats: {"value": ..., "span": ...} and direct value
                if isinstance(value_info, dict):
                    if "value" not in value_info:
                        continue
                    normalized_value_info = {
                        "value": str(value_info["value"]),
                        "span": value_info.get("span", [])
                    }
                else:
                    normalized_value_info = {
                        "value": str(value_info),
                        "span": []
                    }

                normalized_entry[sublabel] = normalized_value_info

            if normalized_entry:
                result[normalized_cat].append(normalized_entry)

    return result

def evaluate_predictions_from_json(model_name, text_json_path, pred_json_path, gold_json_path):
    with open(text_json_path, "r", encoding="utf-8") as f:
        text_data = json.load(f)

    with open(pred_json_path, "r", encoding="utf-8") as f:
        pred_data = json.load(f)

    with open(gold_json_path, "r", encoding="utf-8") as f:
        gold_data = json.load(f)

    # Handle case where gold data is a list
    if isinstance(gold_data, list):
        gold_data = {str(item["id"]): item["gold_output"] for item in gold_data if "id" in item}
    
    # Handle case where text data is a list
    if isinstance(text_data, list):
        text_data = {str(item.get("id", idx)): item for idx, item in enumerate(text_data)}
    
    # Handle case where pred data is a list
    if isinstance(pred_data, list):
        pred_data = {str(item["id"]): item for item in pred_data if "id" in item}

    all_ids = set(text_data.keys()) & set(pred_data.keys()) & set(gold_data.keys())
    if not all_ids:
        raise ValueError("No matching IDs found between text, predictions and gold data")

    ner_metric = evaluate.load("seqeval")
    all_y_true = []
    all_y_pred = []
    results_per_file = []
    stats_lines = []
    comparison_lines = []

    # Output paths
    output_base = "/home/s27mhusa_hpc/Master-Thesis/Evaluation_Results/SentenceLevelResults-30July"
    output_path = f"{output_base}/ner_eval_{model_name}.txt"
    stats_output_path = f"{output_base}/Stats/ner_evaluation_stats_{model_name}.txt"
    comparison_output_path = f"{output_base}/Comparisons/ner_comparison_{model_name}.jsonl"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(stats_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(comparison_output_path), exist_ok=True)

    output_base = "/home/s27mhusa_hpc/Master-Thesis/Evaluation_Results/SentenceLevelResults-30July"
    comparison_dir = os.path.join(output_base, "Comparisons")

    for item_id in all_ids:
        text_entry = text_data.get(item_id, {})
        pred_entry = pred_data.get(item_id, {})
        gold_entry = gold_data.get(item_id, {})

        text = text_entry.get("text", "")
        pred_ann = normalize_pred_entities(pred_entry)
        gold_ann = gold_entry if isinstance(gold_entry, dict) else {}
        # save_comparison_files(pred_ann, gold_ann, text, comparison_dir, item_id)

        try:
            tokens_pred, y_pred, stats = generate_bio_from_json(text, pred_ann)
            tokens_gold, y_true, _ = generate_bio_from_json(text, gold_ann)

            if len(y_pred) != len(y_true):
                print(f"⚠️ Length mismatch for item {item_id}, skipping.")
                continue

            all_y_true.append(y_true)
            all_y_pred.append(y_pred)

            stats_lines.append(f"Prediction stats for item {item_id}:\n{stats}")

            if any(label != "O" for label in y_true + y_pred):
                report = classification_report([y_true], [y_pred])
                results_per_file.append(f"Item {item_id}:\n{report}")

            comparison_lines.append(json.dumps({
                "id": item_id,
                "text": text,
                "gold_entities": gold_ann,
                "predicted_entities": pred_ann,
                "tokens": tokens_gold,
                "gold_BIO": y_true,
                "pred_BIO": y_pred
            }, ensure_ascii=False))

        except Exception as e:
            print(f"❌ Failed to process item {item_id}: {e}")
            continue

    # Compute overall results if we have any predictions
    if all_y_true and all_y_pred:
        overall_results = ner_metric.compute(predictions=all_y_pred, references=all_y_true, zero_division=0)

        print("\n📊 Overall Results:")
        print(classification_report(all_y_true, all_y_pred))
        print(f"Accuracy: {overall_results['overall_accuracy']:.4f}")
        print(f"F1: {overall_results['overall_f1']:.4f}")

        # Save evaluation report
        with open(output_path, "w", encoding="utf-8") as f:
            for line in results_per_file:
                f.write(line + "\n")
            f.write("\n📊 Overall Results:\n")
            f.write(classification_report(all_y_true, all_y_pred))
            f.write(f"\nAccuracy: {overall_results['overall_accuracy']:.4f}")
            f.write(f"\nF1: {overall_results['overall_f1']:.4f}")
    else:
        print("⚠️ No valid predictions to evaluate")

    with open(stats_output_path, "w", encoding="utf-8") as f:
        for line in stats_lines:
            f.write(str(line) + "\n")

    with open(comparison_output_path, "w", encoding="utf-8") as f:
        for line in comparison_lines:
            f.write(line + "\n")

    print(f"📁 Results saved to {output_path}")
    print(f"📁 Comparison file saved to {comparison_output_path}")

    return overall_results if all_y_true and all_y_pred else None

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    args = parser.parse_args()

    evaluate_predictions_from_json(
        model_name=args.model_name,
        text_json_path="/home/s27mhusa_hpc/Master-Thesis/LLM-Predictions-Sentence/Test_ner_dataset_sentence_text_entity.json",
        pred_json_path="/home/s27mhusa_hpc/Master-Thesis/LLM-Predictions-Sentence/Test_predictions_Qwen2.5-72B-Instruct_extracted.json",
        gold_json_path="/home/s27mhusa_hpc/Master-Thesis/LLM-Predictions-Sentence/Test_gold_Qwen2.5-72B-Instruct_extracted.json"
    )