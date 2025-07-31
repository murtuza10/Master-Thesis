from datasets import Dataset, DatasetDict
import json

def convert_chat_jsonl_to_single_text_column(jsonl_path):
    data = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            chat = json.loads(line)
            data.append({"text": chat})

    return Dataset.from_list(data)

# Example usage
train_dataset  = convert_chat_jsonl_to_single_text_column("/home/s27mhusa_hpc/Master-Thesis/dataset_finetune_llm_train_final.jsonl")
val_dataset  = convert_chat_jsonl_to_single_text_column("/home/s27mhusa_hpc/Master-Thesis/dataset_finetune_llm_val_final.jsonl")
# Save locally (optional)


# Combine into a DatasetDict
dataset = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset
})


dataset.save_to_disk("/home/s27mhusa_hpc/Master-Thesis/final_dataset_train")

dataset = dataset.load_from_disk("/home/s27mhusa_hpc/Master-Thesis/final_dataset_train")
print(dataset)
print(dataset["train"].column_names)


