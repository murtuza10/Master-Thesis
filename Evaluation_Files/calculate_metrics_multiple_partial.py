import ast
import json
import os
import evaluate
import argparse
import sys
import spacy
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

# -----------------------------------------------------------------------------

def evaluate_all(model_name, input_text_dir, input_annot_dir, input_annot_dir_json, xmi_dir):
    """
    Evaluate NER predictions across multiple documents with corresponding .xmi files.
    Includes both exact match (seqeval) and partial match evaluation.
    """

    extract_json_block_from_directory(input_annot_dir, input_annot_dir_json, model_name)
    ner_metric = evaluate.load("seqeval")

    all_y_true = []
    all_y_pred = []
    tokens_all = []
    results_per_file = []
    stats_lines = []

    y_true_dir = f"/home/s27mhusa_hpc/Master-Thesis/Test_BIO_labels"
    results_output_path = f"/home/s27mhusa_hpc/Master-Thesis/Evaluation_Results/TestFiles_6thAugust/ner_evaluation_results_{model_name}_2shot_partial.txt"
    stats_output_path = f"/home/s27mhusa_hpc/Master-Thesis/Evaluation_Results/TestFiles_6thAugust/Stats/ner_evaluation_stats_{model_name}_2shot_partial.txt"

    for filename in os.listdir(input_text_dir):
        if filename.endswith("_inception.txt"):
            file_id = filename.replace("_inception.txt", "")
            text_path = os.path.join(input_text_dir, filename)
            annot_path = os.path.join(input_annot_dir_json, f"{file_id}_inception.txt")
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

                # print(f"File: {filename}")
                # print(f"Y_true: {y_true}")
                # print(f"Y_pred: {y_pred}")

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

    # -------------------- Save Results -----------------------------
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

    print(f"\n📁 All results saved to {results_output_path}")
    return results_per_file, overall_results, partial_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NER output against XMI gold standard.")
    parser.add_argument("--model_name", required=True, help="Name of the model to evaluate.")
    args = parser.parse_args()

    model_name = args.model_name
    input_text_dir = "/home/s27mhusa_hpc/Master-Thesis/Text_Files_For_LLM_Input"
    input_annot_dir = f"/home/s27mhusa_hpc/Master-Thesis/Results/Results_Chat_GPT"
    input_annot_dir_json = f"/home/s27mhusa_hpc/Master-Thesis/Results/Results_Chat_GPT_JSON"
    xmi_dir = "/home/s27mhusa_hpc/Master-Thesis/XMI_Files"

    evaluate_all(
        model_name,
        input_text_dir,
        input_annot_dir,
        input_annot_dir_json,
        xmi_dir
    )
