import json

def convert_jsonl_to_chat_format(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            item = json.loads(line)
            user_prompt = {
                "content": f"Extract named entities related to crops, soil, location, and time.\n\nText:\n{item['input']}",
                "role": "user"
            }
            assistant_response = {
                "content": json.dumps(item["output"], ensure_ascii=False),
                "role": "assistant"
            }
            chat_line = user_prompt, assistant_response
            outfile.write(json.dumps(chat_line, ensure_ascii=False) + "\n")

# Example usage


def main():
    
    input_file = "/home/s27mhusa_hpc/Master-Thesis/Dataset-25-July-Document-Level/Combined_ner_dataset_document_input_output.jsonl"
    output_file = "/home/s27mhusa_hpc/Master-Thesis/Dataset-25-July-Document-Level/Combined_ner_dataset_document_chat.jsonl"
    convert_jsonl_to_chat_format(input_file, output_file)

if __name__ == "__main__":
    main()