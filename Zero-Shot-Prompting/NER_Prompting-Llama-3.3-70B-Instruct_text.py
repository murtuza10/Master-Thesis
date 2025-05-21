import os
import json
import torch
import re
import sys
import os
# Go one level up to the parent directory (Master-Thesis)
sys.path.append(os.path.abspath('..'))
from transformers import AutoModelForCausalLM, AutoTokenizer
from Evaluation_Files.generate_ner_prompt import generate_ner_prompts


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

# Step 3: Perform NER with LLaMA
def perform_ner_with_llama(model, tokenizer, text, max_length=1512):
    system_prompt, user_prompt = generate_ner_prompts(text)
    prompt = f"{system_prompt}\n\n{user_prompt}"    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length, padding=True).to("cuda")
    torch.cuda.empty_cache()
    outputs = model.generate(
        **inputs,
        max_new_tokens=8192,
        temperature=0.7,
        top_p=0.9,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


# Step 5: Process Multiple Text Files
def process_text_files(input_dir, model, tokenizer, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            input_text_path = os.path.join(input_dir, filename)
            output_text_path = os.path.join(output_dir, filename.replace(".txt", "_annotated.txt"))

            # Skip if output file already exists
            if os.path.exists(output_text_path):
                print(f"Skipping {filename} (already processed).")
                continue
            
            with open(input_text_path, "r", encoding="utf-8") as file:
                text = file.read()
            
            print(f"Processing {filename}...")
            ner_result = perform_ner_with_qwen(model, tokenizer, text)
            
            with open(output_text_path, "w", encoding="utf-8") as file:
                file.write(ner_result)
            
            print(f"NER results saved to {output_text_path}")
            

# Step 6: Main Execution
if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    input_dir = "/home/s27mhusa_hpc/Master-Thesis/Text_Files_For_LLM_Input"
    output_dir = "/home/s27mhusa_hpc/Master-Thesis/Results/Results_new_prompt/LLM_annotated_Llama-3.3-70B-Instruct"  # Change to the desired output directory path
    local_model_path = "/lustre/scratch/data/s27mhusa_hpc-murtuza_master_thesis/Llama-3.3-70B-Instruct"

    llama_model, llama_tokenizer = load_llama_model(local_model_path)
    process_text_files(input_dir, llama_model, llama_tokenizer, output_dir)

    print("NER processing complete.")
