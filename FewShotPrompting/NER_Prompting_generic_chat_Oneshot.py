import os
import sys
import json
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoConfig

sys.path.append(os.path.abspath('..'))

from Evaluation_Files.generate_ner_prompt_1Example import generate_ner_prompts
from Evaluation_Files.calculate_metrics_multiple_embeddings_excel import evaluate_all



def load_model(model_path):
    print(f"Loading model from: {model_path}")
    
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    
    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Then load model with clean config
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    ).eval()
    print("Successfully loaded in FP16")
    return model, tokenizer
    
def perform_ner(model, tokenizer, text, max_length):
    system_prompt, user_prompt = generate_ner_prompts(text)

    # Build messages list according to role-based chat format
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenize the prompt
    input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to(device)
    output_ids = model.generate(input_ids=input_ids,max_new_tokens=1500,
        temperature=0.7,
        top_p=0.9,)

    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
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
    parser.add_argument("--start", type=int, default=3, help="Starting point for evaluation.")
    parser.add_argument("--max_length", type=int, default=1512, help="Maximum length for tokenized input.")


    args = parser.parse_args()
    model_name = args.model_name

    model, tokenizer = load_model(args.model_path)
    process_text_files(args.input_dir, model, tokenizer, args.output_dir, int(args.max_length))
    # input_text_dir = "/home/s27mhusa_hpc/Master-Thesis/Text_Files_For_LLM_Input"
    # input_annot_dir = f"/home/s27mhusa_hpc/Master-Thesis/Results/Results_new_prompt/LLM_annotated_{model_name}"
    # input_annot_dir_json = f"/home/s27mhusa_hpc/Master-Thesis/Results/Results_new_prompt_json/LLM_annotated_{model_name}"
    log_dir = os.environ.get('LOG_DIR')

    xmi_dir = "/home/s27mhusa_hpc/Master-Thesis/NewDatasets27August/Test_XMI_Files"
    evaluate_all(args.model_name,
        args.input_dir,
        args.output_dir,
        args.output_dir_json,
        int(args.top_k),
        xmi_dir,
        log_dir)
    print("NER processing complete.")


if __name__ == "__main__":
    main()
