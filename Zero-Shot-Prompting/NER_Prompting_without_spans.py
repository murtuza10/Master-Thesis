import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
import json

# Step 1: Load LLaMA Model and Tokenizer
def load_llama_model(model_path):
    print("Loading model and tokenizer from local directory...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype=torch.float16
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Model and tokenizer loaded from local path.")
    return model, tokenizer

# Step 2: Define Prompt for NER
def create_ner_prompt(text):
    prompt = f"""
    Perform Named Entity Recognition (NER) on the text below and extract entities related to Crops and Soils. Return entity spans along with the extracted values.
Provide the results strictly and only in the following JSON format:
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

Text:
{text}
    """
    return prompt


# Step 3: Perform NER with LLaMA
def perform_ner_with_llama(model, tokenizer, text, max_length=4096):
    # Create the NER prompt\
    prompt = create_ner_prompt(text)
    
    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length,padding=True).to("cuda")  # Remove .to("cuda")

    # Generate output
    outputs = model.generate(
        **inputs,
        max_new_tokens=1000,
        temperature=0.7,
        top_p=0.9,
    )
    
    # Decode the output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response



# Step 4: Process Text Data from CSV
def process_csv(csv_path, model, tokenizer, output_csv_path="ner_results.csv"):
    print(f"Loading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)  # Load CSV file
    
    if "abstract_text_2" not in df.columns:
        raise ValueError("CSV must contain a column named 'text' with the text data.")
    
    # Apply NER to each row
    results = []
    for idx, text in enumerate(df["abstract_text_2"]):
        print(f"Processing row {idx + 1}/{len(df)}...")
        try:
            ner_result = perform_ner_with_llama(model, tokenizer, text)
            results.append(ner_result)
        except Exception as e:
            print(f"Error processing row {idx + 1}: {e}")
            results.append("Error")
    
    # Add results to the DataFrame
    df["ner_results"] = results
    
    # Save to a new CSV file
    df.to_csv(output_csv_path, index=False)
    print(f"NER results saved to {output_csv_path}")
    return df

# Step 5: Main Execution
if __name__ == "__main__":
    # Path to the input CSV file 
    input_csv_path = "/home/s27mhusa_hpc/pilot-uc-textmining-metadata/data/Bonares/output/test_data.csv"  
    
    local_model_path = "/lustre/scratch/data/s27mhusa_hpc-ner_model_data/Llama-3.1-8B"  

    # Load the model
    llama_model, llama_tokenizer = load_llama_model(local_model_path)
    
    # Perform NER and save results
    output_csv_path = "/home/s27mhusa_hpc/pilot-uc-textmining-metadata/data/Bonares/output/Llama-3.1-8B_test.csv"
    processed_df = process_csv(input_csv_path, llama_model, llama_tokenizer, output_csv_path)
    
    print("NER processing complete. Sample results:")
    print(processed_df.head())
