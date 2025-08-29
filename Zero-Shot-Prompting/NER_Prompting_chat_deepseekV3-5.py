import requests
import json
import os
import sys
import json
import argparse

sys.path.append(os.path.abspath('..'))

from Evaluation_Files.generate_ner_prompt_5Examples import generate_ner_prompts
from Evaluation_Files.calculate_metrics_multiple import evaluate_all



def perform_ner(text,max_length):
    system_prompt, user_prompt = generate_ner_prompts(text)

    response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": "Bearer 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",
        "Content-Type": "application/json"
    },
    data=json.dumps({
        "model": "deepseek/deepseek-chat-v3-0324",
        "messages": [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
        ],
        "provider": {
            "sort": "price"
        },
        "temperature": 0.7,
        "max_tokens": max_length,
        "top_p": 0.90,
        # "response_format": { "type": "json_object" }
    })
    )
    return response.json()


def process_text_files(input_dir, output_dir,max_length):
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            input_text_path = os.path.join(input_dir, filename)
            output_text_path = os.path.join(output_dir, filename.replace(".txt", "_annotated.txt"))

            if os.path.exists(output_text_path):
                with open(output_text_path, "r", encoding="utf-8") as file:
                    text = file.read()
                    data = json.loads(text)  
                    content = (
                        data.get('choices', [{}])[0]       # Get first choice or default to [{}]
                            .get('message', {})            # Get 'message' dict or empty dict
                            .get('content')                # Get 'content' or None
                    )

                    if content and content.strip():
                        print(f"Skipping {filename} (already processed).")
                        continue

            with open(input_text_path, "r", encoding="utf-8") as file:
                text = file.read()

            print(f"Processing {filename}...")
            ner_result = perform_ner(text,max_length)

            with open(output_text_path, "w", encoding="utf-8") as file:
                json.dump(ner_result, file, ensure_ascii=False, indent=2)

            print(f"Saved: {output_text_path}")


def main():
    parser = argparse.ArgumentParser(description="Perform NER using a transformer-based model.")
    parser.add_argument("--input_dir", required=True, help="Directory containing input text files.")
    parser.add_argument("--output_dir", required=True, help="Directory to save annotated output files.")
    parser.add_argument("--output_dir_json", required=True, help="Directory to save extracted json output from annotated files.")
    parser.add_argument("--max_length", type=int, default=1512, help="Maximum length for tokenized input.")

    args = parser.parse_args()

    process_text_files(args.input_dir, args.output_dir, args.max_length)
    # input_text_dir = "/home/s27mhusa_hpc/Master-Thesis/Text_Files_For_LLM_Input"
    # input_annot_dir = f"/home/s27mhusa_hpc/Master-Thesis/Results/Results_new_prompt/LLM_annotated_{model_name}"
    # input_annot_dir_json = f"/home/s27mhusa_hpc/Master-Thesis/Results/Results_new_prompt_json/LLM_annotated_{model_name}"
    xmi_dir = "/home/s27mhusa_hpc/Master-Thesis/XMI_Files"
    evaluate_all("DeepSeekV3",
        args.input_dir,
        args.output_dir,
        args.output_dir_json,
        5,
        xmi_dir)
    print("NER processing complete.")


if __name__ == "__main__":
    main()




