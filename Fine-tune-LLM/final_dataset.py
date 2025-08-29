import json

def convert_jsonl_to_autotrain_format(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for line in lines:
            try:
                pair = json.loads(line.strip())  # this is a list of user and assistant
                if isinstance(pair, list) and len(pair) == 2:
                    formatted = {"messages": pair}
                    outfile.write(json.dumps(formatted, ensure_ascii=False) + "\n")
                else:
                    print(f"⚠️ Skipping malformed line: {line[:60]}...")
            except json.JSONDecodeError as e:
                print(f"❌ Error decoding JSON: {e}")
    
    print(f"✅ Finished! Converted file written to: {output_file}")


if __name__ == "__main__":
    input_file = "/home/s27mhusa_hpc/Master-Thesis/NewDatasets27August/Test_ner_dataset_sentence_chat.jsonl"
    output_file = "/home/s27mhusa_hpc/Master-Thesis/NewDatasets27August/Test_ner_dataset_sentence_chat_final.jsonl"
    convert_jsonl_to_autotrain_format(input_file, output_file)
