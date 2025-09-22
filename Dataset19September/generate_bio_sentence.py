import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))  # ensures current directory is included
sys.path.append("/home/s27mhusa_hpc/Master-Thesis/Evaluation_Files")
from generate_bio_from_cas_sentence import generate_bio_annotations_from_cas

XMI_DIR = "/home/s27mhusa_hpc/Master-Thesis/Dataset19September/Test_XMI_Files_English"
OUTPUT_DIR_LABELS = "/home/s27mhusa_hpc/Master-Thesis/Dataset19September/Test_BIO_labels_English_Specific_sentence"
OUTPUT_DIR_TOKENS = "/home/s27mhusa_hpc/Master-Thesis/Dataset19September/Test_BIO_tokens_English_Specific_sentence"

if __name__ == "__main__":
    # Create output directories
    os.makedirs(OUTPUT_DIR_LABELS, exist_ok=True)
    os.makedirs(OUTPUT_DIR_TOKENS, exist_ok=True)
    
    for filename in os.listdir(XMI_DIR):
        if filename.endswith('.xmi'):
            print(f"Processing {filename}...")
            xmi_path = os.path.join(XMI_DIR, filename)
            
            try:
                sentences_data = generate_bio_annotations_from_cas(xmi_path)
                
                # Process each sentence separately
                for i, sent_data in enumerate(sentences_data):
                    base_name = filename.replace(".xmi", f"_sentence_{i+1:03d}")
                    
                    # Write labels file
                    with open(os.path.join(OUTPUT_DIR_LABELS, f"{base_name}.txt"), "w", encoding="utf-8") as f: 
                        f.write(str(sent_data['labels']))
                    
                    # Write tokens file
                    with open(os.path.join(OUTPUT_DIR_TOKENS, f"tokens_{base_name}.txt"), "w", encoding="utf-8") as f: 
                        f.write(str(sent_data['tokens']))
                
                print(f"Successfully processed {filename} - {len(sentences_data)} sentences")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    print("Processing complete!")