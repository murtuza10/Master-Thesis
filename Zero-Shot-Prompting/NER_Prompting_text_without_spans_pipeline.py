import os
import torch
from transformers import pipeline

# Step 1: Load LLaMA Model with Pipeline
def load_llama_pipeline(model_path):
    print("Loading model pipeline from local directory...")
    pipe = pipeline("text-generation", model=model_path, device=0)  # Use device=0 for GPU
    print("Pipeline loaded successfully.")
    return pipe

# Step 2: Define System and User Prompts
def get_system_prompt():
    return """
    You are an AI specialized in Named Entity Recognition (NER) for extracting information about Crops and Soils from text. Your task is to analyze the input text and extract relevant entities in JSON format. The entities you need to identify are:

    - **Crops**: Species, Variety, Cultivar  
    - **Soils**: Texture, Depth, Bulk_density, pH_value, Organic_carbon, Available_nitrogen  

    Return the extracted information in the following **exact** JSON format:  

    {
        "Crops": [
            {
                "Species": {"value": "<species_name>"},
                "Variety": {"value": "<variety>"},
                "Cultivar": {"value": "<cultivar>"}
            }
        ],
        "Soils": [
            {
                "Texture": {"value": "<texture>"},
                "Depth": {"value": "<depth>"},
                "Bulk_density": {"value": "<bulk_density>"},
                "pH_value": {"value": "<pH_value>"},
                "Organic_carbon": {"value": "<organic_carbon>"},
                "Available_nitrogen": {"value": "<available_nitrogen>"}
            }
        ]
    }

    Ensure the JSON format is **strictly** followed. If any value is not present, return `null`. Do not add any explanations, just output the JSON result.
    """

def create_user_prompt(text):
    return f"""
    Perform Named Entity Recognition (NER) on the following text:

    "{text}"
    """

# Step 3: Perform NER with LLaMA Pipeline
def perform_ner_with_llama(pipe, text, max_length=512):
    system_prompt = get_system_prompt()
    user_prompt = create_user_prompt(text)
    full_prompt = system_prompt + "\n\n" + user_prompt
    
    torch.cuda.empty_cache()
    outputs = pipe(full_prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.9)
    return outputs[0]['generated_text']

# Step 4: Process Multiple Text Files
def process_text_files(input_dir, pipe, output_dir, limit=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, filename in enumerate(os.listdir(input_dir)):
        if filename.endswith(".txt"):
            input_text_path = os.path.join(input_dir, filename)
            output_text_path = os.path.join(output_dir, filename.replace(".txt", "_annotated.txt"))
            
            with open(input_text_path, "r", encoding="utf-8") as file:
                text = file.read()
            
            print(f"Processing {filename}...")
            ner_result = perform_ner_with_llama(pipe, text)
            
            with open(output_text_path, "w", encoding="utf-8") as file:
                file.write(ner_result)
            
            print(f"NER results saved to {output_text_path}")
            
            if i + 1 == limit:
                break

# Step 5: Main Execution
if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    input_dir = "/home/s27mhusa_hpc/pilot-uc-textmining-metadata/data/Bonares/output/filtered_df_soil_crop_year_LTE_test"
    output_dir = "/home/s27mhusa_hpc/pilot-uc-textmining-metadata/data/Bonares/output/filtered_df_soil_crop_year_LTE_test_annotated-Llama-3.1-8B"
    local_model_path = "/lustre/scratch/data/s27mhusa_hpc-ner_model_data/Llama-3.1-8B"

    llama_pipeline = load_llama_pipeline(local_model_path)
    process_text_files(input_dir, llama_pipeline, output_dir)

    print("NER processing complete.")
