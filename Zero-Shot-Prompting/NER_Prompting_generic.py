import os
import sys
import json
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.append(os.path.abspath('..'))

from Evaluation_Files.generate_ner_prompt import generate_ner_prompts
from Evaluation_Files.calculate_metrics_multiple import evaluate_all


def load_model(model_path):
    print(f"Loading model and tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Model and tokenizer loaded successfully.")
    return model, tokenizer


def perform_ner(model, tokenizer, text, max_length):
    system_prompt, user_prompt = generate_ner_prompts(text)
    prompt = f"{system_prompt}\n\n{user_prompt}"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length, padding=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=1500,
        temperature=0.7,
        top_p=0.9,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def process_text_files(input_dir, model, tokenizer, output_dir, max_length):
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            input_text_path = os.path.join(input_dir, filename)
            output_text_path = os.path.join(output_dir, filename.replace(".txt", "_annotated.txt"))

            if os.path.exists(output_text_path):
                print(f"Skipping {filename} (already processed).")
                continue

            with open(input_text_path, "r", encoding="utf-8") as file:
                text = file.read()

            print(f"Processing {filename}...")
            ner_result = perform_ner(model, tokenizer, text, max_length)

            with open(output_text_path, "w", encoding="utf-8") as file:
                file.write(ner_result)

            print(f"Saved: {output_text_path}")


def main():
    parser = argparse.ArgumentParser(description="Perform NER using a transformer-based model.")
    parser.add_argument("--input_dir", required=True, help="Directory containing input text files.")
    parser.add_argument("--output_dir", required=True, help="Directory to save annotated output files.")
    parser.add_argument("--output_dir_json", required=True, help="Directory to save extracted json output from annotated files.")
    parser.add_argument("--model_name", required=True, help="Name of the model used for annotation.")
    parser.add_argument("--model_path", required=True, help="Path to the pretrained model.")
    parser.add_argument("--max_length", type=int, default=1512, help="Maximum length for tokenized input.")

    args = parser.parse_args()
    model_name = args.model_name

    model, tokenizer = load_model(args.model_path)
    process_text_files(args.input_dir, model, tokenizer, args.output_dir, args.max_length)
    # input_text_dir = "/home/s27mhusa_hpc/Master-Thesis/Text_Files_For_LLM_Input"
    # input_annot_dir = f"/home/s27mhusa_hpc/Master-Thesis/Results/Results_new_prompt/LLM_annotated_{model_name}"
    # input_annot_dir_json = f"/home/s27mhusa_hpc/Master-Thesis/Results/Results_new_prompt_json/LLM_annotated_{model_name}"
    xmi_dir = "/home/s27mhusa_hpc/Master-Thesis/XMI_Files_OpenAgrar"
    evaluate_all(model_name,
        args.input_dir,
        args.output_dir,
        args.output_dir_json,
        xmi_dir)
    print("NER processing complete.")


if __name__ == "__main__":
    main()
