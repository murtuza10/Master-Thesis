import re
import regex
import json
import os

def extract_second_json_block_from_directory(input_dir, output_dir):
    """
    Extracts the second JSON block from each .txt file in the input directory
    and saves it as a pretty-printed JSON file in the output directory.
    
    Args:
        input_dir (str): Path to the input directory containing .txt files.
        output_dir (str): Path to the output directory to save extracted JSON.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Loop through all .txt files in the input directory
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

                # Extract all JSON blocks
                pattern = r'\{(?:[^{}]|(?R))*\}'
                matches = regex.findall(pattern, content)
                # for i,match in enumerate(matches):
                #     print(f"Match {i}: {match}")

                for i in range(1, len(matches)):
                    try:
                        data = json.loads(matches[i])
                        with open(output_file, "w", encoding="utf-8") as out:
                            json.dump(data, out, indent=2)
                        print(f"✅ Processed: {filename} (using block {i + 1})")
                        break  # Stop after the first successful parse
                    except json.JSONDecodeError as je:
                        print(f"⚠️ Block {i + 1} in {filename} is not valid JSON, trying next...")

                else:
                    print(f"⚠️ No valid JSON block found in {filename} after the first one.")
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")

# Example usage:
# extract_second_json_block_from_directory("/path/to/input", "/path/to/output")



if __name__ == "__main__":
    extract_second_json_block_from_directory(
        "/home/s27mhusa_hpc/Master-Thesis/Results/Results_new_prompt/LLM_annotated_Qwen2.5-72B-Instruct",
        "/home/s27mhusa_hpc/Master-Thesis/Results/Results_new_prompt_json/LLM_annotated_Qwen2.5-72B-Instruct"
    )
