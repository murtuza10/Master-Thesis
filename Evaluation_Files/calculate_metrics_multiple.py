import os
import evaluate
import argparse
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))  # ensures current directory is included
sys.path.append("/home/s27mhusa_hpc/Master-Thesis/Evaluation_Files")
from generate_bio_from_cas import generate_bio_annotations_from_cas
from generate_bio_from_json import generate_bio_from_json
from seqeval.metrics import classification_report
import spacy
from extract_json_from_output import extract_second_json_block_from_directory


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

def evaluate_all(model_name, input_text_dir, input_annot_dir, input_annot_dir_json, xmi_dir):
    """
    Evaluate NER predictions across multiple documents with corresponding .xmi files.

    Args:
        input_text_dir (str): Directory with plain text files.
        input_annot_dir (str): Directory with annotation JSON files.
        xmi_dir (str): Directory with ground truth .xmi files.
    """

    extract_second_json_block_from_directory(
    input_annot_dir,
    input_annot_dir_json
    )
    ner_metric = evaluate.load("seqeval")

    all_y_true = []
    all_y_pred = []
    results_per_file = []
    results_output_path = f"/home/s27mhusa_hpc/Master-Thesis/Evaluation_Results/Modified_Prompt/ner_evaluation_results_{model_name}.txt"
    stats_output_path = f"/home/s27mhusa_hpc/Master-Thesis/Evaluation_Results/Modified_Prompt/Stats/ner_evaluation_stats_{model_name}.txt"

    results_lines = []  # Collect output to write to file later
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
                # results_lines.append(msg)                
                continue

            try:
                y_true = generate_bio_annotations_from_cas(xmi_path)

                if os.path.exists(annot_path):
                    token, y_pred, stats = generate_bio_from_json(text_path, annot_path)
                    stats_lines.append(f"{file_id}:\n")
                    stats_lines.append(stats)
                else:
                    msg = f"ℹ️ No annotation file for {filename}, assuming no predictions."
                    print(msg)
                    # results_lines.append(msg)                    
                    y_pred = generate_empty_bio(text_path)

                if len(y_true) != len(y_pred):
                    msg = f"❌ Length mismatch in {file_id} — skipping."
                    print(msg)
                    # results_lines.append(msg)                    
                    continue

                # print(f"File_name:{filename}")
                # print(f"Y_true:{y_true}")
                # print(f"Y_pred:{y_pred}")

                all_y_true.append(y_true)
                all_y_pred.append(y_pred)

                results = ner_metric.compute(predictions=[y_pred], references=[y_true], zero_division=0)
                results_per_file.append(f"{file_id}:\n")
                report = classification_report([y_true], [y_pred])
                results_per_file.append(report)
                results_per_file.append(f"Accuracy: {results['overall_accuracy']}\n")
                print(f"✅ {file_id}: {results['overall_f1']:.4f} F1")
            except Exception as e:
                msg = f"❌ Error processing {file_id}: {e}"
                print(msg)
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
    parser = argparse.ArgumentParser(description="Evaluate NER output against XMI gold standard.")
    parser.add_argument("--model_name", required=True, help="Name of the model to evaluate.")
    args = parser.parse_args()

    model_name = args.model_name
    input_text_dir = "/home/s27mhusa_hpc/Master-Thesis/Text_Files_For_LLM_Input"
    input_annot_dir = f"/home/s27mhusa_hpc/Master-Thesis/Results/Results_modified_prompt/LLM_annotated_{model_name}"
    input_annot_dir_json = f"/home/s27mhusa_hpc/Master-Thesis/Results/Results_modified_prompt_json/LLM_annotated_{model_name}"
    xmi_dir = "/home/s27mhusa_hpc/Master-Thesis/XMI_Files"

    evaluate_all(
        model_name,
        input_text_dir,
        input_annot_dir,
        input_annot_dir_json,
        xmi_dir
    )
