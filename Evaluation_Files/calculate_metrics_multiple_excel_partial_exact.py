import ast
import json
import os
import evaluate
import argparse
import sys
import spacy
import time
import pandas as pd
from datetime import datetime
from seqeval.metrics import classification_report

# Add local module paths
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append("/home/s27mhusa_hpc/Master-Thesis/Evaluation_Files")

from generate_bio_from_cas import generate_bio_annotations_from_cas
from generate_bio_from_json import generate_bio_from_json
from extract_json_from_output import extract_json_block_from_directory


def has_named_entities(labels):
    return any(label.startswith("B-") or label.startswith("I-") for label in labels)

def generate_empty_bio(text_path):
    """Generate 'O' tags equal to token count using spaCy whitespace tokenization."""
    nlp = spacy.load("en_core_web_sm")
    with open(text_path, 'r', encoding='utf-8') as f:
        doc = nlp(f.read())
        tokens = [token.text for token in doc]
        labels = ["O"] * len(tokens)
    return tokens, labels

# ----------------------- NEW FUNCTIONS FOR PARTIAL MATCH -----------------------

def bio_to_spans(tokens, labels):
    """Convert BIO labels to spans: (start_index, end_index, label)."""
    spans = []
    start = None
    label = None
    for i, tag in enumerate(labels):
        if tag.startswith("B-"):
            if start is not None:
                spans.append((start, i, label))
            start = i
            label = tag[2:]
        elif tag.startswith("I-") and label == tag[2:]:
            continue
        else:
            if start is not None:
                spans.append((start, i, label))
                start = None
                label = None
    if start is not None:
        spans.append((start, len(labels), label))
    return spans

def partial_match(pred_spans, true_spans):
    """
    Partial match: counts overlap in token indices for same label as a TP.
    """
    tp = 0
    used = set()
    for ps in pred_spans:
        for j, ts in enumerate(true_spans):
            if j in used:
                continue
            # Check overlap in token positions and label match
            if ps[1] > ts[0] and ts[1] > ps[0] and ps[2] == ts[2]:
                tp += 1
                used.add(j)
                break
    fp = len(pred_spans) - tp
    fn = len(true_spans) - tp
    return tp, fp, fn

def evaluate_partial(all_tokens, all_y_true, all_y_pred):
    """Partial match evaluation across all docs."""
    total_tp = total_fp = total_fn = 0
    for tokens, y_true, y_pred in zip(all_tokens, all_y_true, all_y_pred):
        true_spans = bio_to_spans(tokens, y_true)
        pred_spans = bio_to_spans(tokens, y_pred)
        tp, fp, fn = partial_match(pred_spans, true_spans)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": total_tp, "fp": total_fp, "fn": total_fn}

def get_power_consumption(log_dir=None):
    """Calculate power consumption from GPU logs."""
    if not log_dir:
        return 0.0
        
    gpu_log_file = os.path.join(log_dir, "gpu_usage.log")
    power_file = os.path.join(log_dir, "power_consumption.txt")
    
    if os.path.exists(gpu_log_file):
        try:
            # Run the power calculator
            import subprocess
            result = subprocess.run([
                'python', 
                '/home/s27mhusa_hpc/Master-Thesis/Power-Consumption/gpu_power_calculator.py',
                gpu_log_file
            ], capture_output=True, text=True, cwd=log_dir)
            
            if result.returncode == 0:
                # Try to extract power from output
                import re
                match = re.search(r'(\d+\.?\d*)\s*kWh', result.stdout)
                if match:
                    power_val = float(match.group(1))
                    # Save it for future use
                    with open(power_file, 'w') as f:
                        f.write(f"{power_val} kWh")
                    return power_val
        except Exception as e:
            print(f"Warning: Could not calculate power consumption: {e}")

    return 0.0

def create_excel_data(model_name, overall_results, report, partial_results, power_consumption, result_link, stats_link, shot_count=None):
    """
    Create data structure matching the Excel format with both exact and partial match results.
    """
    model_display_name = f"{model_name}_{shot_count}_shot" if shot_count is not None else model_name

    excel_row = {
        'Date': datetime.now().strftime('%m/%d/%Y'),
        'Model Name': model_name,
        
        # Exact Match Results (Micro averages from seqeval)
        'Exact Precision (Micro Avg)': report['micro avg']['precision'],
        'Exact Recall (Micro Avg)': report['micro avg']['recall'],
        'Exact F1 Score (Micro Avg)': report['micro avg']['f1-score'],

        # Exact Match Results (Macro averages from sklearn classification report)
        'Exact Precision (Macro Avg)': report['macro avg']['precision'],
        'Exact Recall (Macro Avg)': report['macro avg']['recall'],
        'Exact F1 Score (Macro Avg)': report['macro avg']['f1-score'],
        
        # Exact Match Results (Weighted averages from sklearn classification report)
        'Exact Precision (Weighted Avg)': report['weighted avg']['precision'],
        'Exact Recall (Weighted Avg)': report['weighted avg']['recall'],
        'Exact F1 Score (Weighted Avg)': report['weighted avg']['f1-score'],
        
        # Partial Match Results
        'Partial Precision': partial_results['precision'],
        'Partial Recall': partial_results['recall'],
        'Partial F1 Score': partial_results['f1'],
        'Partial TP': partial_results['tp'],
        'Partial FP': partial_results['fp'],
        'Partial FN': partial_results['fn'],
        
        'Support': report['weighted avg']['support'],
        'Accuracy': overall_results.get('overall_accuracy', 0.0),
        
        'Result Link': result_link,
        'Stats Link': stats_link,
        
        'No of GPU Used': '4 MLGPU',  # Adjust based on your setup
        'Power Consumption': f'{power_consumption:.3f} kWh',
    }
    
    return excel_row

def save_to_excel(excel_data, excel_output_path):
    """
    Save the data to Excel file, appending if file exists.
    """
    try:
        # Check if file exists
        if os.path.exists(excel_output_path):
            # Read existing data
            existing_df = pd.read_excel(excel_output_path)
            # Append new row
            new_df = pd.concat([existing_df, pd.DataFrame([excel_data])], ignore_index=True)
        else:
            # Create new DataFrame
            new_df = pd.DataFrame([excel_data])
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(excel_output_path), exist_ok=True)
        
        # Save to Excel
        new_df.to_excel(excel_output_path, index=False)
        
    except Exception as e:
        print(f"❌ Error saving to Excel: {e}")

# -----------------------------------------------------------------------------

def evaluate_all(model_name, input_text_dir, input_annot_dir, input_annot_dir_json, start, xmi_dir, log_dir):
    """
    Evaluate NER predictions across multiple documents with corresponding .xmi files.
    Includes both exact match (seqeval) and partial match evaluation.
    """

    extract_json_block_from_directory(input_annot_dir, input_annot_dir_json, model_name,start)
    ner_metric = evaluate.load("seqeval")

    all_y_true = []
    all_y_pred = []
    tokens_all = []
    results_per_file = []
    stats_lines = []

    y_true_dir = f"/home/s27mhusa_hpc/Master-Thesis/NewDatasets27August/Test_BIO_labels"
    results_output_path = f"/home/s27mhusa_hpc/Master-Thesis/Evaluation_Results/Final_TestFiles_5thSeptember_FewShotTest_Specific/ner_evaluation_results_{model_name}_{start}_shot.txt"
    stats_output_path = f"/home/s27mhusa_hpc/Master-Thesis/Evaluation_Results/Final_TestFiles_5thSeptember_FewShotTest_Specific/Stats/ner_evaluation_stats_{model_name}_{start}_shot.txt"

    # Excel output path
    excel_output_path = f"/home/s27mhusa_hpc/Master-Thesis/Evaluation_Results/Final_TestFiles_5thSeptember_FewShotTest_Specific/ner_evaluation_results_{model_name}_{start}_shot.xlsx"

    for filename in os.listdir(input_text_dir):
        if filename.endswith("_inception.txt"):
            file_id = filename.replace("_inception.txt", "")
            text_path = os.path.join(input_text_dir, filename)
            annot_path = os.path.join(input_annot_dir_json, f"{file_id}_inception_annotated.txt")
            xmi_path = os.path.join(xmi_dir, f"{file_id}.xmi")

            if not os.path.exists(xmi_path):
                print(f"⚠️ XMI file not found for {filename}")
                continue

            try:
                y_true_path = os.path.join(y_true_dir, f"{file_id}.txt")
                if os.path.exists(y_true_path):
                    with open(y_true_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        y_true = ast.literal_eval(content)
                    tokens = []  # if saved previously without tokens
                else:
                    tokens, y_true = generate_bio_annotations_from_cas(xmi_path)
                    os.makedirs(os.path.dirname(y_true_path), exist_ok=True)
                    with open(y_true_path, "w", encoding="utf-8") as f:
                        f.write(str(y_true))

                if os.path.exists(annot_path):
                    token, y_pred, stats = generate_bio_from_json(text_path, annot_path)
                    stats_lines.append(f"{file_id}:\n{stats}")
                else:
                    print(f"ℹ️ No annotation file for {filename}, assuming no predictions.")
                    token, y_pred = generate_empty_bio(text_path)

                if len(y_true) != len(y_pred):
                    print(f"❌ Length mismatch in {file_id} — skipping.")
                    continue

                all_y_true.append(y_true)
                all_y_pred.append(y_pred)
                tokens_all.append(token if token else tokens)

                results = ner_metric.compute(predictions=[y_pred], references=[y_true], zero_division=0)
                results_per_file.append(f"{file_id}:\n{classification_report([y_true],[y_pred])}")
                results_per_file.append(f"Accuracy: {results['overall_accuracy']}\n")
                print(f"✅ {file_id}: {results['overall_f1']:.4f} F1 (Exact)")
            except Exception as e:
                print(f"❌ Error processing {file_id}: {e}")

    # -------------------- Exact Match (SeqEval) --------------------
    overall_results = ner_metric.compute(predictions=all_y_pred, references=all_y_true, zero_division=0)
    print("\n📊 Exact Match Overall Performance:")
    print(overall_results)

    # -------------------- Partial Match ----------------------------
    partial_results = evaluate_partial(tokens_all, all_y_true, all_y_pred)
    print("\n📊 Partial Match Overall Performance:")
    print(partial_results)

    # -------------------- Get Classification Report for Excel ------
    report_dict = classification_report(all_y_true, all_y_pred, output_dict=True)
    
    # -------------------- Calculate Power Consumption --------------
    power_consumption = get_power_consumption(log_dir)

    # -------------------- Create and Save Excel Data ---------------
    excel_data = create_excel_data(
        model_name, 
        overall_results, 
        report_dict, 
        partial_results,
        power_consumption,
        results_output_path,
        stats_output_path,
        start
    )
    
    # Save to Excel
    save_to_excel(excel_data, excel_output_path)

    # -------------------- Save Text Results ------------------------
    os.makedirs(os.path.dirname(results_output_path), exist_ok=True)
    with open(results_output_path, "w", encoding="utf-8") as f:
        for line in results_per_file:
            f.write(str(line) + "\n")
        f.write("\n📊 Exact Match Overall Performance:\n")
        f.write(str(classification_report(all_y_true, all_y_pred)) + "\n")
        f.write(f"Overall Accuracy: {overall_results['overall_accuracy']}\n")
        f.write("\n📊 Partial Match Overall Performance:\n")
        f.write(str(partial_results) + "\n")

    os.makedirs(os.path.dirname(stats_output_path), exist_ok=True)
    with open(stats_output_path, "w", encoding="utf-8") as f:
        for line in stats_lines:
            f.write(str(line) + "\n")

    print(f"\n📁 Text results saved to {results_output_path}")
    print(f"📊 Excel results saved to {excel_output_path}")
    print(f"🔋 Power consumption: {power_consumption:.3f} kWh")
    
    return results_per_file, overall_results, partial_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NER output against XMI gold standard.")
    parser.add_argument("--model_name", required=True, help="Name of the model to evaluate.")
    parser.add_argument("--shot_count", type=int, help="Number of shots used for few-shot learning.")
    parser.add_argument("--log_dir", help="Directory containing GPU usage logs for power calculation.")
    args = parser.parse_args()

    model_name = args.model_name
    shot_count = args.shot_count
    log_dir = args.log_dir
    
    input_text_dir = "/home/s27mhusa_hpc/Master-Thesis/Text_Files_For_LLM_Input"
    input_annot_dir = f"/home/s27mhusa_hpc/Master-Thesis/Results/Results_Chat_GPT"
    input_annot_dir_json = f"/home/s27mhusa_hpc/Master-Thesis/Results/Results_Chat_GPT_JSON"
    xmi_dir = "/home/s27mhusa_hpc/Master-Thesis/XMI_Files"

    evaluate_all(
        model_name,
        input_text_dir,
        input_annot_dir,
        input_annot_dir_json,
        xmi_dir,
        log_dir,
        shot_count
    )