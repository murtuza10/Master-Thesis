import evaluate
from generate_bio_from_cas import generate_bio_annotations_from_cas
from generate_bio_from_json import generate_bio_from_json


ner_metric = evaluate.load("seqeval")
y_true = generate_bio_annotations_from_cas('rieglerh.xmi')
print(f"y_true = {y_true}")
input_file_text = "/home/s27mhusa_hpc/pilot-uc-textmining-metadata/data/Bonares/output/Text_Files_From_Inception/00bee634-47e6-490b-89ba-2464c9f09c31_inception.txt"
input_file_annotations = "/home/s27mhusa_hpc/pilot-uc-textmining-metadata/data/Bonares/output/Results_new_prompt_json/filtered_df_soil_crop_year_LTE_test_annotated_Qwen2.5-7B-Instruct/00bee634-47e6-490b-89ba-2464c9f09c31_annotated.txt"
y_pred = generate_bio_from_json(input_file_text,input_file_annotations)
# y_true.insert(0, 'O')
print(f"y_pred = {y_pred}")


from seqeval.metrics.sequence_labeling import get_entities

# Check what entities are being detected
# print("Entities in y_true:", get_entities(y_true))
# print("Entities in y_pred:", get_entities(y_pred))


results = ner_metric.compute(predictions=[y_pred], references=[y_true], zero_division=1)
print(results)
