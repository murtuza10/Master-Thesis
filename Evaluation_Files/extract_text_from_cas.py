from cassis import *
import spacy
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))  # ensures current directory is included
sys.path.append("/home/s27mhusa_hpc/Master-Thesis/Evaluation_Files")

nlp = spacy.load("en_core_web_sm")



def cas_to_text(cas_file):
    with open('/home/s27mhusa_hpc/Master-Thesis/Evaluation_Files/TypeSystem.xml', 'rb') as f:
        typesystem = load_typesystem(f)

    with open(cas_file, 'rb') as f:
        cas = load_cas_from_xmi(f, typesystem=typesystem)
    text = cas.sofa_string
    return text

if __name__ == "__main__":
    XMI_DIR = "/home/s27mhusa_hpc/Master-Thesis/XMI_Files_OpenAgrar"
    OUTPUT_DIR = "/home/s27mhusa_hpc/Master-Thesis/Text_Files_OpenAgrar"
    for filename in os.listdir(XMI_DIR):
        xmi_path = os.path.join(XMI_DIR, filename)
        text = cas_to_text(xmi_path)
        with open(os.path.join(OUTPUT_DIR, filename.replace(".xmi","_inception.txt")), "w", encoding="utf-8") as f: 
            f.write(str(text))
