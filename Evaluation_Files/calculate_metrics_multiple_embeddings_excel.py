import ast
import json
import os
import evaluate
import argparse
import sys
import pandas as pd
from datetime import datetime
import time

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append("/home/s27mhusa_hpc/Master-Thesis/Evaluation_Files")
from generate_bio_from_cas import generate_bio_annotations_from_cas
from generate_bio_from_json import generate_bio_from_json
from seqeval.metrics import classification_report
import spacy
from extract_json_from_output import extract_json_block_from_directory


def has_named_entities(labels):
    return any(label.startswith("B-") or label.startswith("I-") for label in labels)


def generate_empty_bio(text_path):
    """
    Generate a list of 'O' tags equal to the number of tokens in the text.
    Assumes whitespace tokenization.
    """
    nlp = spacy.load("en_core_web_sm")

    with open(text_path, 'r', encoding='utf-8') as f:
        doc = nlp(f.read())
        tokens = [token.text for token in doc]
        labels = ["O"] * len(tokens)
    return labels


def calculate_per_class_metrics(all_y_true, all_y_pred):
    """
    Calculate per-class precision, recall, and F1 scores.
    Returns a dictionary with class-wise metrics.
    """
    from sklearn.metrics import classification_report as sklearn_report
    
    # Flatten the lists for sklearn
    y_true_flat = [label for doc in all_y_true for label in doc]
    y_pred_flat = [label for doc in all_y_pred for label in doc]
    
    # Get unique labels
    all_labels = sorted(list(set(y_true_flat + y_pred_flat)))
    
    # Calculate metrics using sklearn
    report_dict = sklearn_report(y_true_flat, y_pred_flat, 
                                labels=all_labels, 
                                output_dict=True, 
                                zero_division=0)
    # print(report_dict)
    return report_dict


def evaluate_all(model_name, input_text_dir, input_annot_dir, input_annot_dir_json, start, xmi_dir, log_dir):
    """
    Evaluate NER predictions across multiple documents with corresponding .xmi files.
    """
    start_time = time.time()
    start = int(start)

    extract_json_block_from_directory(
        input_annot_dir,
        input_annot_dir_json,
        model_name, start
    )
    ner_metric = evaluate.load("seqeval")

    all_y_true = []
    all_y_pred = []
    results_per_file = []
    y_true_dir = f"/home/s27mhusa_hpc/Master-Thesis/NewDatasets27August/Test_BIO_labels"
    results_output_path = f"/home/s27mhusa_hpc/Master-Thesis/Evaluation_Results/Final_TestFiles_29thAugust_FewShotTests/ner_evaluation_results_{model_name}_{start}_shot.txt"
    stats_output_path = f"/home/s27mhusa_hpc/Master-Thesis/Evaluation_Results/Final_TestFiles_29thAugust_FewShotTests/Stats/ner_evaluation_stats_{model_name}_{start}_shot.txt"

    # Excel output path
    excel_output_path = f"/home/s27mhusa_hpc/Master-Thesis/Evaluation_Results/Final_TestFiles_29thAugust_FewShotTests/ner_evaluation_results_{model_name}_{start}_shot.xlsx"

    results_lines = []
    stats_lines = []

    for filename in os.listdir(input_text_dir):
        if filename.endswith("_inception.txt"):
            file_id = filename.replace("_inception.txt", "")
            text_path = os.path.join(input_text_dir, filename)
            annot_path = os.path.join(input_annot_dir_json, f"{file_id}_inception_annotated.txt")
            xmi_path = os.path.join(xmi_dir, f"{file_id}.xmi")

            if not os.path.exists(xmi_path):
                msg = f"⚠️ XMI file not found for {filename}"
                print(msg)
                continue

            try:
                y_true_path = os.path.join(y_true_dir, f"{file_id}.txt")
                if os.path.exists(y_true_path):
                    with open(y_true_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        y_true = ast.literal_eval(content)
                else:
                    tokens, y_true = generate_bio_annotations_from_cas(xmi_path)
                    os.makedirs(os.path.dirname(y_true_path), exist_ok=True)
                    with open(y_true_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(y_true))
                
                if os.path.exists(annot_path):
                    token, y_pred, stats = generate_bio_from_json(text_path, annot_path)
                    stats_lines.append(f"{file_id}:\n")
                    stats_lines.append(stats)
                else:
                    msg = f"ℹ️ No annotation file for {filename}, assuming no predictions."
                    print(msg)
                    y_pred = generate_empty_bio(text_path)

                if len(y_true) != len(y_pred):
                    msg = f"❌ Length mismatch in {file_id} — skipping."
                    print(msg)
                    continue

                # print(f"File_name:{filename}")
                # print(f"Y_true:{y_true}")
                # print(f"Y_pred:{y_pred}")

                all_y_true.append(y_true)
                all_y_pred.append(y_pred)

                results = ner_metric.compute(predictions=[y_pred], references=[y_true], zero_division=0)
                results_per_file.append(f"{file_id}:\n")
                has_true_entities = any(l != "O" for l in y_true)
                has_pred_entities = any(l != "O" for l in y_pred)
                if (has_true_entities or has_pred_entities):
                    report = classification_report([y_true], [y_pred])
                    results_per_file.append(report)
                else:
                    results_per_file.append(results)
                results_per_file.append(f"Accuracy: {results['overall_accuracy']}\n")
                print(f"✅ {file_id}: {results['overall_f1']:.4f} F1")
            except Exception as e:
                msg = f"❌ Error processing {file_id}: {e}"
                print(msg)

    # print("\n📊 Overall Performance:")
    overall_results = ner_metric.compute(predictions=all_y_pred, references=all_y_true, zero_division=0)
    # print(overall_results)


    
    # Calculate execution time
    end_time = time.time()
    execution_time = end_time - start_time
    power_consumption = get_power_consumption(log_dir)
    
   

    # Save text results (keeping original functionality)
    results_lines.append("\n📊 Overall Performance:")
    report_dict = classification_report(all_y_true, all_y_pred,output_dict=True)
    report = classification_report(all_y_true, all_y_pred)
    results_lines.append(report)

     # Create Excel data
    excel_data = create_excel_data(
        model_name, 
        overall_results, 
        report_dict, 
        power_consumption,
        results_output_path,
        stats_output_path
    )
    
    # Save to Excel
    save_to_excel(excel_data, excel_output_path)


   
    results_lines.append(f"Overall Accuracy: {overall_results['overall_accuracy']}")

    os.makedirs(os.path.dirname(results_output_path), exist_ok=True)

    with open(results_output_path, "w", encoding="utf-8") as f:
        for line in results_per_file:
            f.write(str(line) + "\n")
        for line in results_lines:
            f.write(str(line) + "\n")
    
    os.makedirs(os.path.dirname(stats_output_path), exist_ok=True)

    with open(stats_output_path, "w", encoding="utf-8") as f:
        for line in stats_lines:
            f.write(str(line) + "\n")

    print(f"\n📁 Text results saved to {results_output_path}")
    print(f"📊 Excel results saved to {excel_output_path}")

    return results_per_file, overall_results

def get_power_consumption(log_dir=None):
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



def create_excel_data(model_name, overall_results, report, power_consumption, result_link, stats_link):
    """
    Create data structure matching the Excel format with only micro/macro/weighted averages.
    """

    excel_row = {
        'Date': datetime.now().strftime('%m/%d/%Y'),
        'Model Name': model_name,
        
        # Micro averages (from seqeval)
        'Precision (Micro Avg)': report['micro avg']['precision'],
        'Recall (Micro Avg)': report['micro avg']['recall'],
        'F1 Score (Micro Avg)': report['micro avg']['f1-score'],

        # Macro averages (from sklearn classification report)
        'Precision (Macro Avg)': report['macro avg']['precision'],
        'Recall (Macro Avg)': report['macro avg']['recall'],
        'F1 Score (Macro Avg)': report['macro avg']['f1-score'],
        
        # Weighted averages (from sklearn classification report)
        'Precision (Weighted Avg)': report['weighted avg']['precision'],
        'Recall (Weighted Avg)': report['weighted avg']['recall'],
        'F1 Score (Weighted Avg)': report['weighted avg']['f1-score'],
        'Support': report['weighted avg']['support'],
        
        'Accuracy': overall_results.get('overall_accuracy', 0.0),
        
        'Result Link': result_link,
        'Stats Link': stats_link,
        
        'No of GPU Used': '4 SGPU',  # Adjust based on your setup
        'Power Consumption': f'{power_consumption:.3f} kWh',  # Estimate based on time
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NER output against XMI gold standard.")
    parser.add_argument("--model_name", required=True, help="Name of the model to evaluate.")
    args = parser.parse_args()

    model_name = args.model_name
    input_text_dir = "/home/s27mhusa_hpc/Master-Thesis/Text_Files_Test_Data"
    input_annot_dir = f"/home/s27mhusa_hpc/Master-Thesis/Results/FinalResults_TestFiles_27thAugust_Embeddings/LLM_annotated_{model_name}_5shot"
    input_annot_dir_json = f"/home/s27mhusa_hpc/Master-Thesis/Results/FinalResults_TestFiles_27thAugust_Embeddings_json/LLM_annotated_{model_name}_5shot"
    xmi_dir = "/home/s27mhusa_hpc/Master-Thesis/NewDatasets27August/Test_XMI_Files"
    log_dir = "/home/s27mhusa_hpc/Master-Thesis/FinalOutput-27thAugust-Embeddings/job_monitor_logs_TestFilesQwen2.5-7B-Instruct_20250827_160535_5shot"

    evaluate_all(args.model_name,
        input_text_dir,
        input_annot_dir,
        input_annot_dir_json,
        5,
        xmi_dir,
        log_dir)
