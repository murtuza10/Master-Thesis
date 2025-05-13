import os
import json
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

# Step 1: Load LLaMA Model and Tokenizer
def load_llama_model(model_path):
    print("Loading model and tokenizer from local directory...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="auto",
    torch_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype=torch.float16
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Model and tokenizer loaded from local path.")
    return model, tokenizer

# Step 2: Define Prompt for NER
def generate_ner_prompts(text):
    system_prompt = """
    You are an expert in Named Entity Recognition (NER) for agricultural texts, specializing in identifying crop and soil-related entities. Your task is to extract relevant entities from the provided text and return them strictly in JSON format.

    Follow these rules:
    
    1. Extract entities related to Crops(cropSpecies, cropVariety, cropCultivar) and Soil(soilTexture, soilDepth, soilBulkDensity, soilPH, soilOrganicCarbon, soilAvailableNitrogen). 
    2. Return the results **strictly** in the JSON format shown below.
    3. Include character spans **[start_index, end_index]** for each extracted entity in the complete text.
    4. If no entities are found in a category, return an empty list (e.g., \"Crops\": []).
    5. If the distinction between a cultivar and a variety is unclear, classify it as a variety.
    6. For all entities also annotate the mention of the entity name, for eg annotate {"soilPH": {"value":"pH", "span": [25, 26]}}.
    7. Do **not** include any explanations, extra text, or formatting outside the JSON.

    JSON format:
    {"Crops": [{"": {"value": "", "span": [start_index, end_index]}
                }],
     "Soil": [{"": {"value": "", "span": [start_index, end_index]}
                }]
    }
    """
    
    user_prompt = f"""
    Perform Named Entity Recognition (NER) on the text below and extract entities related to Crops and Soils.
    
    Text:
    {text}
    """
    
    return system_prompt.strip(), user_prompt.strip()
# Step 3: Perform NER with LLaMA
def perform_ner_with_llama(model, tokenizer, text, max_length=1024):
    system_prompt, user_prompt = generate_ner_prompts(text)
    prompt = f"{system_prompt}\n\n{user_prompt}"    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length, padding=True).to("cuda")
    torch.cuda.empty_cache()
    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,
        temperature=0.7,
        top_p=0.9,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


# Step 5: Process Multiple Text Files
def process_text_files(input_dir, model, tokenizer, output_dir):
    i = 0
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            input_text_path = os.path.join(input_dir, filename)
            output_text_path = os.path.join(output_dir, filename.replace(".txt", "_annotated.txt"))
            
            with open(input_text_path, "r", encoding="utf-8") as file:
                text = file.read()
            
            print(f"Processing {filename}...")
            ner_result = perform_ner_with_llama(model, tokenizer, text)
            
            with open(output_text_path, "w", encoding="utf-8") as file:
                file.write(ner_result)
            
            print(f"NER results saved to {output_text_path}")
            i = i+1
            if i==10:
                break

# Step 6: Main Execution
if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    input_dir = "/home/s27mhusa_hpc/pilot-uc-textmining-metadata/data/Bonares/output/filtered_df_soil_crop_year_LTE_test"
    output_dir = "/home/s27mhusa_hpc/pilot-uc-textmining-metadata/data/Bonares/output/Results/filtered_df_soil_crop_year_LTE_test_Llama-3.1-8B-Instruct"  # Change to the desired output directory path
    local_model_path = "/lustre/scratch/data/s27mhusa_hpc-ner_model_data/Llama-3.1-8B-Instruct"

    llama_model, llama_tokenizer = load_llama_model(local_model_path)
    process_text_files(input_dir, llama_model, llama_tokenizer, output_dir)

    print("NER processing complete.")
