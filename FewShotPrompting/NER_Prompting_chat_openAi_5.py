import requests
import json
import os
import sys
import json
import argparse

sys.path.append(os.path.abspath('..'))

from Evaluation_Files.generate_ner_prompt_5Examples import generate_ner_prompts
from Evaluation_Files.calculate_metrics_multiple_excel_partial_exact_count import evaluate_all



from dotenv import load_dotenv
import os
import time
import random
load_dotenv()


def perform_ner(text, max_length, max_retries=5, base_delay=1):
    system_prompt, user_prompt = generate_ner_prompts(text)
    
    api_key = os.getenv("OPENAI_API_KEY")
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url="https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                data=json.dumps({
                    "model": "gpt-5",
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
                    "max_completion_tokens": max_length,
                    "reasoning_effort": "minimal",
                })
            )
            
            # Check if the request was successful
            if response.status_code == 200:
                try:
                    return response.json()
                except json.JSONDecodeError:
                    print(f"Failed to parse JSON response. Raw response: {response.text}")
                    raise
            
            # Handle rate limiting (status code 429)
            elif response.status_code == 429:
                if attempt < max_retries - 1:
                    # Extract retry-after header if available
                    retry_after = response.headers.get('retry-after')
                    if retry_after:
                        delay = int(retry_after)
                        print(f"Rate limited. Waiting {delay} seconds before retry {attempt + 1}/{max_retries}")
                    else:
                        # Exponential backoff with jitter
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        print(f"Rate limited. Waiting {delay:.2f} seconds before retry {attempt + 1}/{max_retries}")
                    
                    time.sleep(delay)
                    continue
                else:
                    print(f"Max retries ({max_retries}) reached for rate limiting")
                    response.raise_for_status()
            
            # Handle other HTTP errors
            else:
                print(f"API Error - Status Code: {response.status_code}")
                print(f"Response Text: {response.text}")
                response.raise_for_status()
                
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"Request failed: {e}. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
                continue
            else:
                print(f"Max retries ({max_retries}) reached for request errors")
                raise
    
    raise Exception(f"Failed to complete request after {max_retries} attempts")


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
    parser.add_argument("--start", type=int, default=3, help="Starting point for evaluation.")


    args = parser.parse_args()

    process_text_files(args.input_dir, args.output_dir, args.max_length)
    # input_text_dir = "/home/s27mhusa_hpc/Master-Thesis/Text_Files_For_LLM_Input"
    # input_annot_dir = f"/home/s27mhusa_hpc/Master-Thesis/Results/Results_new_prompt/LLM_annotated_{model_name}"
    # input_annot_dir_json = f"/home/s27mhusa_hpc/Master-Thesis/Results/Results_new_prompt_json/LLM_annotated_{model_name}"
    log_dir = os.environ.get('LOG_DIR')

    xmi_dir = "/home/s27mhusa_hpc/Master-Thesis/NewDatasets27August/Test_XMI_Files"
    evaluate_all("gpt-5",
        args.input_dir,
        args.output_dir,
        args.output_dir_json,
        int(args.start),
        xmi_dir,
        log_dir)
    print("NER processing complete.")


if __name__ == "__main__":
    main()




