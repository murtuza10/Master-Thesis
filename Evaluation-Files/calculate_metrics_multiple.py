import os
import evaluate
from generate_bio_from_cas import generate_bio_annotations_from_cas
from generate_bio_from_json import generate_bio_from_json
from seqeval.metrics import classification_report

def evaluate_all(input_text_dir, input_annot_dir, xmi_dir):
    """
    Evaluate NER predictions across multiple documents with corresponding .xmi files.

    Args:
        input_text_dir (str): Directory with plain text files.
        input_annot_dir (str): Directory with annotation JSON files.
        xmi_dir (str): Directory with ground truth .xmi files.
    """
    ner_metric = evaluate.load("seqeval")

    all_y_true = []
    all_y_pred = []
    results_per_file = []

    for filename in os.listdir(input_text_dir):
        if filename.endswith("_inception.txt"):
            file_id = filename.replace("_inception.txt", "")
            text_path = os.path.join(input_text_dir, filename)
            annot_path = os.path.join(input_annot_dir, f"{file_id}_annotated.txt")
            xmi_path = os.path.join(xmi_dir, f"{file_id}.xmi")

            if not os.path.exists(annot_path):
                print(f"⚠️ Annotation file not found for {filename}")
                continue
            if not os.path.exists(xmi_path):
                print(f"⚠️ XMI file not found for {filename}")
                continue

            try:
                y_true = generate_bio_annotations_from_cas(xmi_path)
                y_pred = generate_bio_from_json(text_path, annot_path)

                if len(y_true) != len(y_pred):
                    print(f"❌ Length mismatch in {file_id} — skipping.")
                    continue

                # Store for global evaluation
                all_y_true.append(y_true)
                all_y_pred.append(y_pred)

                # Per-file metric
                results = ner_metric.compute(predictions=[y_pred], references=[y_true], zero_division=1)
                results_per_file.append((file_id, results))
                print(f"✅ {file_id}: {results['overall_f1']:.4f} F1")
            except Exception as e:
                print(f"❌ Error processing {file_id}: {e}")

    print("\n📊 Overall Performance:")
    overall_results = ner_metric.compute(predictions=all_y_pred, references=all_y_true, zero_division=1)
    print(overall_results)

    return results_per_file, overall_results

# Example usage
evaluate_all(
    input_text_dir="/home/s27mhusa_hpc/pilot-uc-textmining-metadata/data/Bonares/output/Text_Files_From_Inception",
    input_annot_dir="/home/s27mhusa_hpc/pilot-uc-textmining-metadata/data/Bonares/output/Results_new_prompt_json/filtered_df_soil_crop_year_LTE_test_annotated_Qwen2.5-7B-Instruct",
    xmi_dir="/home/s27mhusa_hpc/pilot-uc-textmining-metadata/data/Bonares/output/XMI_Files"
)
