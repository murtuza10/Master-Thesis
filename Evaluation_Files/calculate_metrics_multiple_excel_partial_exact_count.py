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
from collections import defaultdict, Counter

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
    Returns detailed per-category statistics.
    """
    tp_by_category = defaultdict(int)
    fp_by_category = defaultdict(int)
    fn_by_category = defaultdict(int)
    
    used = set()
    
    # Count true positives
    for ps in pred_spans:
        matched = False
        for j, ts in enumerate(true_spans):
            if j in used:
                continue
            # Check overlap in token positions and label match
            if ps[1] > ts[0] and ts[1] > ps[0] and ps[2] == ts[2]:
                tp_by_category[ps[2]] += 1
                used.add(j)
                matched = True
                break
        if not matched:
            fp_by_category[ps[2]] += 1
    
    # Count false negatives
    for j, ts in enumerate(true_spans):
        if j not in used:
            fn_by_category[ts[2]] += 1
    
    return tp_by_category, fp_by_category, fn_by_category

def exact_match_per_category(pred_spans, true_spans):
    """
    Exact match: entities must match exactly in position and label.
    Returns detailed per-category statistics.
    """
    tp_by_category = defaultdict(int)
    fp_by_category = defaultdict(int)
    fn_by_category = defaultdict(int)
    
    # Convert spans to sets for easy comparison
    pred_set = set(pred_spans)
    true_set = set(true_spans)
    
    # True positives: exact matches
    for span in pred_set & true_set:
        tp_by_category[span[2]] += 1
    
    # False positives: predicted but not in true
    for span in pred_set - true_set:
        fp_by_category[span[2]] += 1
    
    # False negatives: true but not predicted
    for span in true_set - pred_set:
        fn_by_category[span[2]] += 1
    
    return tp_by_category, fp_by_category, fn_by_category

def calculate_metrics_per_category(tp_dict, fp_dict, fn_dict):
    """Calculate precision, recall, and F1 for each category."""
    all_categories = set(tp_dict.keys()) | set(fp_dict.keys()) | set(fn_dict.keys())
    metrics = {}
    
    for category in all_categories:
        tp = tp_dict.get(category, 0)
        fp = fp_dict.get(category, 0)
        fn = fn_dict.get(category, 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics[category] = {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': tp + fn  # Total true entities in this category
        }
    
    return metrics

def evaluate_partial(all_tokens, all_y_true, all_y_pred):
    """Partial match evaluation across all docs with per-category breakdown."""
    total_tp_by_category = defaultdict(int)
    total_fp_by_category = defaultdict(int)
    total_fn_by_category = defaultdict(int)
    
    for tokens, y_true, y_pred in zip(all_tokens, all_y_true, all_y_pred):
        true_spans = bio_to_spans(tokens, y_true)
        pred_spans = bio_to_spans(tokens, y_pred)
        
        tp_dict, fp_dict, fn_dict = partial_match(pred_spans, true_spans)
        
        for category in tp_dict:
            total_tp_by_category[category] += tp_dict[category]
        for category in fp_dict:
            total_fp_by_category[category] += fp_dict[category]
        for category in fn_dict:
            total_fn_by_category[category] += fn_dict[category]
    
    # Calculate per-category metrics
    category_metrics = calculate_metrics_per_category(
        total_tp_by_category, total_fp_by_category, total_fn_by_category
    )
    
    # Calculate overall metrics
    total_tp = sum(total_tp_by_category.values())
    total_fp = sum(total_fp_by_category.values())
    total_fn = sum(total_fn_by_category.values())
    
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    
    return {
        "overall": {
            "precision": overall_precision, 
            "recall": overall_recall, 
            "f1": overall_f1,
            "tp": total_tp, 
            "fp": total_fp, 
            "fn": total_fn
        },
        "by_category": category_metrics
    }

def evaluate_exact_match(all_tokens, all_y_true, all_y_pred):
    """Exact match evaluation across all docs with per-category breakdown."""
    total_tp_by_category = defaultdict(int)
    total_fp_by_category = defaultdict(int)
    total_fn_by_category = defaultdict(int)
    
    for tokens, y_true, y_pred in zip(all_tokens, all_y_true, all_y_pred):
        true_spans = bio_to_spans(tokens, y_true)
        pred_spans = bio_to_spans(tokens, y_pred)
        
        tp_dict, fp_dict, fn_dict = exact_match_per_category(pred_spans, true_spans)
        
        for category in tp_dict:
            total_tp_by_category[category] += tp_dict[category]
        for category in fp_dict:
            total_fp_by_category[category] += fp_dict[category]
        for category in fn_dict:
            total_fn_by_category[category] += fn_dict[category]
    
    # Calculate per-category metrics
    category_metrics = calculate_metrics_per_category(
        total_tp_by_category, total_fp_by_category, total_fn_by_category
    )
    
    return {"by_category": category_metrics}

def format_category_results(category_results, match_type=""):
    """Format per-category results for display and saving."""
    lines = []
    lines.append(f"\n{'='*50}")
    lines.append(f"{match_type} Per-Category Entity Analysis")
    lines.append(f"{'='*50}")
    
    if not category_results:
        lines.append("No entities found.")
        return lines
    
    # Header
    lines.append(f"{'Category':<15} {'TP':<6} {'FP':<6} {'FN':<6} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}")
    lines.append("-" * 80)
    
    # Sort categories alphabetically
    for category in sorted(category_results.keys()):
        metrics = category_results[category]
        lines.append(
            f"{category:<15} "
            f"{metrics['tp']:<6} "
            f"{metrics['fp']:<6} "
            f"{metrics['fn']:<6} "
            f"{metrics['precision']:<10.4f} "
            f"{metrics['recall']:<10.4f} "
            f"{metrics['f1']:<10.4f} "
            f"{metrics['support']:<10}"
        )
    
    return lines

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

def create_excel_data(model_name, overall_results, report, partial_results, exact_category_results, power_consumption, result_link, stats_link, shot_count=None):
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
        'Partial Precision': partial_results['overall']['precision'],
        'Partial Recall': partial_results['overall']['recall'],
        'Partial F1 Score': partial_results['overall']['f1'],
        'Partial TP': partial_results['overall']['tp'],
        'Partial FP': partial_results['overall']['fp'],
        'Partial FN': partial_results['overall']['fn'],
        
        'Support': report['weighted avg']['support'],
        'Accuracy': overall_results.get('overall_accuracy', 0.0),
        
        'Result Link': result_link,
        'Stats Link': stats_link,
        
        'No of GPU Used': '4 MLGPU',  # Adjust based on your setup
        'Power Consumption': f'{power_consumption:.3f} kWh',
    }
    
    return excel_row

def create_detailed_excel_data(model_name, exact_category_results, partial_results, shot_count=None):
    """Create detailed Excel data with per-category breakdowns."""
    detailed_data = []
    
    # Get all unique categories from both exact and partial results
    all_categories = set()
    if exact_category_results and 'by_category' in exact_category_results:
        all_categories.update(exact_category_results['by_category'].keys())
    if 'by_category' in partial_results:
        all_categories.update(partial_results['by_category'].keys())
    
    for category in sorted(all_categories):
        row = {
            'Date': datetime.now().strftime('%m/%d/%Y'),
            'Model Name': model_name,
            'Shot Count': shot_count,
            'Category': category,
        }
        
        # Exact match data
        if exact_category_results and 'by_category' in exact_category_results and category in exact_category_results['by_category']:
            exact_metrics = exact_category_results['by_category'][category]
            row.update({
                'Exact TP': exact_metrics['tp'],
                'Exact FP': exact_metrics['fp'],
                'Exact FN': exact_metrics['fn'],
                'Exact Precision': exact_metrics['precision'],
                'Exact Recall': exact_metrics['recall'],
                'Exact F1': exact_metrics['f1'],
                'Exact Support': exact_metrics['support'],
            })
        else:
            row.update({
                'Exact TP': 0, 'Exact FP': 0, 'Exact FN': 0,
                'Exact Precision': 0.0, 'Exact Recall': 0.0, 'Exact F1': 0.0,
                'Exact Support': 0,
            })
        
        # Partial match data
        if category in partial_results['by_category']:
            partial_metrics = partial_results['by_category'][category]
            row.update({
                'Partial TP': partial_metrics['tp'],
                'Partial FP': partial_metrics['fp'],
                'Partial FN': partial_metrics['fn'],
                'Partial Precision': partial_metrics['precision'],
                'Partial Recall': partial_metrics['recall'],
                'Partial F1': partial_metrics['f1'],
                'Partial Support': partial_metrics['support'],
            })
        else:
            row.update({
                'Partial TP': 0, 'Partial FP': 0, 'Partial FN': 0,
                'Partial Precision': 0.0, 'Partial Recall': 0.0, 'Partial F1': 0.0,
                'Partial Support': 0,
            })
        
        detailed_data.append(row)
    
    return detailed_data

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

def save_detailed_excel(detailed_data, excel_output_path):
    """Save detailed per-category data to Excel."""
    try:
        # Check if file exists
        if os.path.exists(excel_output_path):
            # Read existing data
            existing_df = pd.read_excel(excel_output_path)
            # Append new rows
            new_df = pd.concat([existing_df, pd.DataFrame(detailed_data)], ignore_index=True)
        else:
            # Create new DataFrame
            new_df = pd.DataFrame(detailed_data)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(excel_output_path), exist_ok=True)
        
        # Save to Excel
        new_df.to_excel(excel_output_path, index=False)
        
        print(f"📊 Detailed category results saved to {excel_output_path}")
        
    except Exception as e:
        print(f"❌ Error saving detailed Excel: {e}")

# -----------------------------------------------------------------------------

def evaluate_all(model_name, input_text_dir, input_annot_dir, input_annot_dir_json, start, xmi_dir, log_dir):
    """
    Evaluate NER predictions across multiple documents with corresponding .xmi files.
    Includes both exact match (seqeval) and partial match evaluation with per-category analysis.
    """

    extract_json_block_from_directory(input_annot_dir, input_annot_dir_json, model_name,start)
    ner_metric = evaluate.load("seqeval")

    all_y_true = []
    all_y_pred = []
    tokens_all = []
    results_per_file = []
    stats_lines = []

    y_true_dir = f"/home/s27mhusa_hpc/Master-Thesis/Dataset1stSeptemberDocumentLevel/Test_BIO_labels"
    results_output_path = f"/home/s27mhusa_hpc/Master-Thesis/Evaluation_Results/Final_TestFiles_22September_FewShotTest_Embeddings_Broad/ner_evaluation_results_{model_name}_{start}_shot.txt"
    stats_output_path = f"/home/s27mhusa_hpc/Master-Thesis/Evaluation_Results/Final_TestFiles_22September_FewShotTest_Embeddings_Broad/Stats/ner_evaluation_stats_{model_name}_{start}_shot.txt"

    # Excel output paths
    excel_output_path = f"/home/s27mhusa_hpc/Master-Thesis/Evaluation_Results/Final_TestFiles_22September_FewShotTest_Embeddings_Broad/ner_evaluation_results_{model_name}_{start}_shot.xlsx"
    detailed_excel_output_path = f"/home/s27mhusa_hpc/Master-Thesis/Evaluation_Results/Final_TestFiles_22September_FewShotTest_Embeddings_Broad/detailed_category_results_{model_name}_{start}_shot.xlsx"

    for filename in os.listdir(input_text_dir):
        if filename.endswith("_inception.txt"):
            file_id = filename.replace("_inception.txt", "")
            text_path = os.path.join(input_text_dir, filename)
            annot_path = os.path.join(input_annot_dir_json, f"{file_id}_inception_annotated.txt")
            xmi_path = os.path.join(xmi_dir, f"{file_id}.xmi")

            if not os.path.exists(xmi_path):
                print(f"⚠️ XMI file not found for {xmi_path}")
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
                    all_y_true.append(y_true)
                    token, y_pred = generate_empty_bio(text_path)
                    all_y_pred.append(y_pred)
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
                all_y_true.append(y_true)
                token, y_pred = generate_empty_bio(text_path)
                all_y_pred.append(y_pred)


    # -------------------- Exact Match (SeqEval) --------------------
    overall_results = ner_metric.compute(predictions=all_y_pred, references=all_y_true, zero_division=0)
    print("\n📊 Exact Match Overall Performance:")
    print(overall_results)

    # -------------------- Exact Match Per-Category Analysis ---------
    exact_category_results = evaluate_exact_match(tokens_all, all_y_true, all_y_pred)
    exact_category_lines = format_category_results(exact_category_results['by_category'], "EXACT MATCH")
    
    print("\n".join(exact_category_lines))

    # -------------------- Partial Match ----------------------------
    partial_results = evaluate_partial(tokens_all, all_y_true, all_y_pred)
    print("\n📊 Partial Match Overall Performance:")
    print(partial_results['overall'])
    
    partial_category_lines = format_category_results(partial_results['by_category'], "PARTIAL MATCH")
    print("\n".join(partial_category_lines))

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
        exact_category_results,
        power_consumption,
        results_output_path,
        stats_output_path,
        start
    )
    
    # Save summary to Excel
    save_to_excel(excel_data, excel_output_path)
    
    # Create and save detailed category data
    detailed_excel_data = create_detailed_excel_data(model_name, exact_category_results, partial_results, start)
    save_detailed_excel(detailed_excel_data, detailed_excel_output_path)

    # -------------------- Save Text Results ------------------------
    os.makedirs(os.path.dirname(results_output_path), exist_ok=True)
    with open(results_output_path, "w", encoding="utf-8") as f:
        for line in results_per_file:
            f.write(str(line) + "\n")
        f.write("\n📊 Exact Match Overall Performance:\n")
        f.write(str(classification_report(all_y_true, all_y_pred)) + "\n")
        f.write(f"Overall Accuracy: {overall_results['overall_accuracy']}\n")
        
        # Add exact match per-category results
        f.write("\n".join(exact_category_lines) + "\n")
        
        f.write("\n📊 Partial Match Overall Performance:\n")
        f.write(str(partial_results['overall']) + "\n")
        
        # Add partial match per-category results
        f.write("\n".join(partial_category_lines) + "\n")

    os.makedirs(os.path.dirname(stats_output_path), exist_ok=True)
    with open(stats_output_path, "w", encoding="utf-8") as f:
        for line in stats_lines:
            f.write(str(line) + "\n")

    print(f"\n📁 Text results saved to {results_output_path}")
    print(f"📊 Excel results saved to {excel_output_path}")
    print(f"📋 Detailed category results saved to {detailed_excel_output_path}")
    print(f"🔋 Power consumption: {power_consumption:.3f} kWh")
    
    return results_per_file, overall_results, partial_results, exact_category_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NER output against XMI gold standard.")
    parser.add_argument("--model_name", required=True, help="Name of the model to evaluate.")
    parser.add_argument("--shot_count", type=int, help="Number of shots used for few-shot learning.")
    args = parser.parse_args()

    model_name = args.model_name
    shot_count = args.shot_count
    log_dir = "/home/s27mhusa_hpc/Master-Thesis/Logs/Llama-3.3-70B"

    os.makedirs(log_dir, exist_ok=True)

    input_text_dir = "/home/s27mhusa_hpc/Master-Thesis/Text_Files_Test_Data"
    input_annot_dir = f"/home/s27mhusa_hpc/Master-Thesis/Results/Corrected/LLM_annotated_Llama-3.3-70B-Instruct_{shot_count}shot"
    input_annot_dir_json = f"/home/s27mhusa_hpc/Master-Thesis/Results/Corrected_json/LLM_annotated_Llama-3.3-70B-Instruct_{shot_count}shot"
    xmi_dir = "/home/s27mhusa_hpc/Master-Thesis/Dataset1stSeptemberDocumentLevel/Test_XMI_Files"

    evaluate_all(
        model_name,
        input_text_dir,
        input_annot_dir,
        input_annot_dir_json,
        shot_count,
        xmi_dir,
        log_dir)