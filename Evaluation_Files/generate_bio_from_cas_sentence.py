from cassis import *
import spacy
import os

nlp = spacy.load("en_core_web_sm")

def cas_to_bio_by_sentences(cas, annotation_types):
    text = cas.sofa_string
    doc = nlp(text)
    
    # Get sentence annotations from CAS
    sentence_annotations = list(cas.select("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"))
    
    sentences_data = []
    
    for sent_ann in sentence_annotations:
        sent_begin = sent_ann.begin
        sent_end = sent_ann.end
        sent_text = text[sent_begin:sent_end]
        
        # Create a spacy doc for this sentence to get tokens
        sent_doc = nlp(sent_text)
        sent_tokens = [token.text for token in sent_doc]
        sent_labels = ["O"] * len(sent_tokens)
        
        # Process annotations for this sentence
        for type_name in annotation_types:
            for ann in cas.select(type_name):
                ann_begin = ann.begin
                ann_end = ann.end
                
                # Skip annotations that don't overlap with this sentence
                if ann_end <= sent_begin or ann_begin >= sent_end:
                    continue
                
                # Get label name based on available features
                label = None
                for feat in ann.type.all_features:
                    if feat.name not in ["sofa", "begin", "end"] and hasattr(ann, feat.name):
                        val = getattr(ann, feat.name)
                        if val:
                            label = val
                            break
                
                if not label:
                    label = type_name.split(".")[-1]  # fallback to type name
                
                # Adjust annotation positions to be relative to sentence start
                rel_ann_begin = max(0, ann_begin - sent_begin)
                rel_ann_end = min(len(sent_text), ann_end - sent_begin)
                
                # Mark tokens within the annotation span
                first_token_in_annotation = True
                for i, token in enumerate(sent_doc):
                    token_start = token.idx
                    token_end = token.idx + len(token)
                    
                    # Check if token overlaps with annotation (relative to sentence)
                    if token_start < rel_ann_end and token_end > rel_ann_begin:
                        # Determine if this is the beginning of the entity
                        if first_token_in_annotation or sent_labels[i-1] == "O" or not sent_labels[i-1].endswith(f"-{label}"):
                            sent_labels[i] = f"B-{label}"
                            first_token_in_annotation = False
                        else:
                            sent_labels[i] = f"I-{label}"
        
        sentences_data.append({
            'tokens': sent_tokens,
            'labels': sent_labels,
            'text': sent_text,
            'start_char': sent_begin,
            'end_char': sent_end,
            'sentence_id': sent_ann.xmi_id if hasattr(sent_ann, 'xmi_id') else len(sentences_data)
        })
    
    return sentences_data



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
    
    sentences_data = cas_to_bio_by_sentences(cas, annotation_types)
    
    return sentences_data

# Alternative function that returns the original format (flat lists)
def generate_bio_annotations_from_cas_flat(cas_file):
    """
    Returns flattened tokens and labels (original format) but processes by CAS sentences
    """
    sentences_data = generate_bio_annotations_from_cas(cas_file)
    
    all_tokens = []
    all_labels = []
    
    for sent_data in sentences_data:
        all_tokens.extend(sent_data['tokens'])
        all_labels.extend(sent_data['labels'])
    
    return all_tokens, all_labels