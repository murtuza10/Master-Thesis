
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))  # ensures current directory is included
sys.path.append("/home/s27mhusa_hpc/Master-Thesis/Evaluation_Files")
from generate_bio_from_cas_timestatement import generate_bio_annotations_from_cas

XMI_DIR = "/home/s27mhusa_hpc/Master-Thesis/Test_Rule_Based_Annotations_8thSeptember"
OUTPUT_DIR_LABELS = "/home/s27mhusa_hpc/Master-Thesis/RuleBasedAnnotations8thSeptember/Rule_BIO_labels_timestatement"
OUTPUT_DIR_TOKENS = "/home/s27mhusa_hpc/Master-Thesis/RuleBasedAnnotations8thSeptember/Rule_BIO_tokens_timestatement"
if __name__ == "__main__":
    for filename in os.listdir(XMI_DIR):
        xmi_path = os.path.join(XMI_DIR, filename)
        tokens, y_true = generate_bio_annotations_from_cas(xmi_path)
        with open(os.path.join(OUTPUT_DIR_LABELS, filename.replace(".xmi",".txt")), "w", encoding="utf-8") as f: 
            f.write(str(y_true))
        with open(os.path.join(OUTPUT_DIR_TOKENS, f"tokens_{filename.replace('.xmi','.txt')}"), "w", encoding="utf-8") as f: 
            f.write(str(tokens))