import re
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
                with open(input_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # Extract all JSON blocks
                matches = re.findall(r'```json\s*({.*?})\s*```', content, re.DOTALL)

                if len(matches) >= 2:
                    data = json.loads(matches[1])  # Second JSON block
                    with open(output_file, "w", encoding="utf-8") as out:
                        json.dump(data, out, indent=2)
                    print(f"✅ Processed: {filename}")
                else:
                    print(f"⚠️ Skipped {filename}: Less than 2 JSON blocks")

            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")

# Example usage:
# extract_second_json_block_from_directory("/path/to/input", "/path/to/output")


extract_second_json_block_from_directory(
    "/home/s27mhusa_hpc/pilot-uc-textmining-metadata/data/Bonares/output/Results_new_prompt/filtered_df_soil_crop_year_LTE_test_annotated_Qwen2.5-7B-Instruct",
    "/home/s27mhusa_hpc/pilot-uc-textmining-metadata/data/Bonares/output/Results_new_prompt_json/filtered_df_soil_crop_year_LTE_test_annotated_Qwen2.5-7B-Instruct"
)
