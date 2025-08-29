import os
import evaluate
import argparse
import ast

import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))  # ensures current directory is included
sys.path.append("/home/s27mhusa_hpc/Master-Thesis/Evaluation_Files")
from generate_bio_from_cas import generate_bio_annotations_from_cas
from seqeval.metrics import classification_report
import spacy


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



def has_named_entities(labels):
    return any(label.startswith("B-") or label.startswith("I-") for label in labels)

def evaluate_all(rule_based_dir, xmi_dir):

    ner_metric = evaluate.load("seqeval")

    all_y_true = []
    all_y_pred = []
    results_per_file = []
    y_true_dir = f"/home/s27mhusa_hpc/Master-Thesis/Test_BIO_labels"
    results_output_path = f"/home/s27mhusa_hpc/Master-Thesis/Evaluation_Results/RuleBased_22August/ner_evaluation_results_rule_based_specific_location.txt"
    stats_output_path = f"/home/s27mhusa_hpc/Master-Thesis/Evaluation_Results/RuleBased_22August/Stats/ner_evaluation_stats_rule_based_specific_location.txt"

    results_lines = []  # Collect output to write to file later
    stats_lines = []


    for filename in os.listdir(xmi_dir):
        filename = filename.replace(".xmi", "")
        xmi_path = os.path.join(xmi_dir, f"{filename}.xmi")
        rule_based_path = os.path.join(rule_based_dir, f"{filename}_inception_inception.xmi")

        if not os.path.exists(xmi_path):
            msg = f"⚠️ XMI file not found for {filename}"
            print(msg)
            # results_lines.append(msg)                
            continue

        try:
            y_true_path = os.path.join(y_true_dir, f"{filename}.txt")
            if os.path.exists(y_true_path):
                # File exists: read and convert to list
                with open(y_true_path, "r", encoding="utf-8") as f:
                    # y_true = f.read().splitlines()
                    content = f.read().strip()
                    y_true = ast.literal_eval(content)
            else:
                # File doesn't exist: generate and save
                tokens,y_true = generate_bio_annotations_from_cas(xmi_path)
                
                # Ensure the directory exists
                os.makedirs(os.path.dirname(y_true_path), exist_ok=True)
                
                # Save the output to the file
                with open(y_true_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(y_true))
            # print(f"Y_true for {filename} is {y_true}")
            # print(f"Y_true is {y_true} and length is {len(y_true)}")
            if os.path.exists(rule_based_path):
                tokens,y_pred = generate_bio_annotations_from_cas(rule_based_path)
                # print(f"Y_pred for {filename} is {y_pred}")

            else:
                msg = f"ℹ️ No annotation file for {filename}, assuming no predictions."
                continue

            if len(y_true) != len(y_pred):
                msg = f"❌ Length mismatch in {filename} — skipping."
                print(msg + f"Length of y_true is {len(y_true)} and length of y_pred is {len(y_pred)}")
                # results_lines.append(msg)                    
                continue

            # print(f"File_name:{filename}")
            # print(f"Y_true:{y_true}")
            # print(f"Y_pred:{y_pred}")

            all_y_true.append(y_true)
            all_y_pred.append(y_pred)

            results = ner_metric.compute(predictions=[y_pred], references=[y_true], zero_division=0)
            results_per_file.append(f"{filename}:\n")
            has_true_entities = any(l != "O" for l in y_true)
            has_pred_entities = any(l != "O" for l in y_pred)
            if (has_true_entities or has_pred_entities):
                report = classification_report([y_true], [y_pred])
                results_per_file.append(report)
            else:
                results_per_file.append(results)
            results_per_file.append(f"Accuracy: {results['overall_accuracy']}\n")
            print(f"✅ {filename}: {results['overall_f1']:.4f} F1")
        except Exception as e:
            msg = f"❌ Error processing {filename}: {e}"
            print(f"Y_true:{y_true}")
            print(f"Y_pred:{y_pred}")
            print(msg)
            print(results)
            # results_lines.append(msg)

    print("\n📊 Overall Performance:")
    overall_results = ner_metric.compute(predictions=all_y_pred, references=all_y_true, zero_division=0)
    print(overall_results)


    results_lines.append("\n📊 Overall Performance:")
    report = classification_report(all_y_true, all_y_pred)
    results_lines.append(report)
    results_lines.append(f"Overall Accuracy: {overall_results['overall_accuracy']}")

    # Save results to file
    with open(results_output_path, "w", encoding="utf-8") as f:
        for line in results_per_file:
            f.write(str(line) + "\n")
        for line in results_lines:
            f.write(str(line) + "\n")
    
    with open(stats_output_path, "w", encoding="utf-8") as f:
        for line in stats_lines:
            f.write(str(line) + "\n")

    print(f"\n📁 All results saved to {results_output_path}")

    return results_per_file, overall_results


if __name__ == "__main__":
    
    xmi_dir = "/home/s27mhusa_hpc/Master-Thesis/XMI_Files"
    rule_based_dir = "/home/s27mhusa_hpc/Master-Thesis/Test_Rule_Based_Annotations"

    evaluate_all(
        rule_based_dir,
        xmi_dir
    )
