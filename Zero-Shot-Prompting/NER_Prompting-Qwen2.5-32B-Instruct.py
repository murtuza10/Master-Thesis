import os
import json
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from generate_ner_prompt import generate_ner_prompts


# Step 1: Load Qwen 2.5-32B Model and Tokenizer
def load_qwen_model(model_path):
    print("Loading Qwen 2.5-32B model and tokenizer from local directory...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Qwen 2.5-32B model and tokenizer loaded from local path.")
    return model, tokenizer


# Step 3: Perform NER with Qwen 2.5-32B
def perform_ner_with_qwen(model, tokenizer, text, max_length=1024):
    system_prompt, user_prompt = generate_ner_prompts(text)
    prompt = f"{system_prompt}\n\n{user_prompt}"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length, padding=True).to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=1500,
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
            ner_result = perform_ner_with_qwen(model, tokenizer, text)
            
            with open(output_text_path, "w", encoding="utf-8") as file:
                file.write(ner_result)
            
            print(f"NER results saved to {output_text_path}")
            i = i+1
            if i==10:
                break

# Step 6: Main Execution
if __name__ == "__main__":
    input_dir = "/home/s27mhusa_hpc/pilot-uc-textmining-metadata/data/Bonares/output/filtered_df_soil_crop_year_LTE_test"
    output_dir = "/home/s27mhusa_hpc/pilot-uc-textmining-metadata/data/Bonares/output/Results/filtered_df_soil_crop_year_LTE_test_annotated_Qwen2.5-32B-Instruct"  # Change to the desired output directory path
    local_model_path = "/lustre/scratch/data/s27mhusa_hpc-ner_model_data/Qwen2.5-32B-Instruct"

    qwen_model, qwen_tokenizer = load_qwen_model(local_model_path)
    process_text_files(input_dir, qwen_model, qwen_tokenizer, output_dir)

    print("NER processing complete.")