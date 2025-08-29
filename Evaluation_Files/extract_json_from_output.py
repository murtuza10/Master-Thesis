import re
import regex
import json
import os

def extract_json_block_from_directory(input_dir, output_dir, model_name, start):
    import os, json

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename)

            try:    
                if os.path.exists(output_file):
                    print(f"Skipping {output_file} (already processed).")
                    continue

                with open(input_file, "r", encoding="utf-8") as f:
                    content = f.read()

                if model_name == "DeepSeekV3":
                    data = json.loads(content)  
                    content = data['choices'][0]['message']['content']

                # --- Extract JSON block between ```json and ``` ---
                cleaned_content = content.strip()
                
                # Find the start of JSON block
                json_start = cleaned_content.find("```json")
                if json_start == -1:
                    print(f"⚠️  No ```json found in {filename}")
                    continue
                
                # Move past the ```json marker
                json_start += 7  # len("```json")
                
                # Find the closing ``` after the json block
                json_end = cleaned_content.find("```", json_start)
                if json_end == -1:
                    print(f"⚠️  No closing ``` found in {filename}")
                    continue
                
                # Extract the JSON block
                json_block = cleaned_content[json_start:json_end].strip()
                
                # Save the extracted JSON block
                with open(output_file, "w", encoding="utf-8") as out:
                    out.write(json_block)

                print(f"✅ Processed: {filename}")

            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
# Example usage:
# extract_second_json_block_from_directory("/path/to/input", "/path/to/output")



if __name__ == "__main__":
    extract_second_json_block_from_directory(
        "/home/s27mhusa_hpc/Master-Thesis/Results/Results_new_prompt/LLM_annotated_Qwen2.5-72B-Instruct",
        "/home/s27mhusa_hpc/Master-Thesis/Results/Results_new_prompt_json/LLM_annotated_Qwen2.5-72B-Instruct"
    )
