from cassis import *
import spacy

nlp = spacy.load("en_core_web_sm")

def cas_to_bio(cas, annotation_types):
    text = cas.sofa_string
    doc = nlp(text)
    tokens = [token.text for token in doc]
    labels = ["O"] * len(tokens)

    for type_name in annotation_types:
        for ann in cas.select(type_name):
            begin = ann.begin
            end = ann.end

            # Get label name based on available features
            label = None
            for feat in ann.type.all_features:
                if feat.name != "sofa" and feat.name != "begin" and feat.name != "end" and hasattr(ann, feat.name):
                    val = getattr(ann, feat.name)
                    if val:
                        label = val
                        break

            if not label or type_name == "webanno.custom.Timestatement":
                label = type_name.split(".")[-1]  # fallback to type name
            elif  type_name == "webanno.custom.Location":
                label = "locationName"

            for i, token in enumerate(doc):
                if token.idx >= begin and (token.idx + len(token)) <= end:
                    labels[i] = f"B-{label}" if token.idx == begin else f"I-{label}"

    return tokens, labels

def generate_bio_annotations_from_cas(cas_file):
    with open('/home/s27mhusa_hpc/Master-Thesis/Evaluation_Files/TypeSystem.xml', 'rb') as f:
        typesystem = load_typesystem(f)

    with open(cas_file, 'rb') as f:
        cas = load_cas_from_xmi(f, typesystem=typesystem)

    annotation_types = [
        "webanno.custom.Crops",
        "webanno.custom.Location",
        "webanno.custom.Soil",
        "webanno.custom.Timestatement",
    ]
    tokens, bio_labels = cas_to_bio(cas, annotation_types)
    # for token, label in zip(tokens, bio_labels):
    #   print(f"{token}: {label}")
    return tokens, bio_labels



if __name__ == "__main__":
    generate_bio_annotations_from_cas('rieglerh.xmi')
