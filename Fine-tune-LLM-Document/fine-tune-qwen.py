# -*- coding: utf-8 -*-
"""
Created on Sun Sep  7 18:40:31 2025

@author: xinxin
"""

from datasets import load_dataset

dataset = load_dataset("json", data_files="/home/s27mhusa_hpc/Master-Thesis/Fine-tune-LLM-Document/text2icasa_training_data_1to1_only_method_fertilizer.jsonl",split='train')

split_dataset = dataset.train_test_split(test_size=2, seed=42)


# You can now access your splits like this:
train_dataset = split_dataset['train']
test_dataset = split_dataset['test']


#pip install transformers datasets peft bitsandbytes accelerate
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/lustre/scratch/data/s27mhusa_hpc-murtuza_master_thesis/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, load_in_8bit=True)

from peft import get_peft_model, LoraConfig
peft_config = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.1)
model = get_peft_model(model, peft_config)

from unsloth import FastLanguageModel
# Load model in 4-bit quantization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=2048,
    load_in_4bit=True
)

from transformers import TrainingArguments
training_arguments = TrainingArguments(
    output_dir="/home/s27mhusa_hpc/Master-Thesis/Fine-tune-LLM-Document/fine_tuned_qwen",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    num_train_epochs=1,
    eval_steps=0.2,
    logging_steps=1,
    warmup_steps=10,
    logging_strategy="steps",
    learning_rate=2e-4,
    fp16=False,
    bf16=False,
    group_by_length=True
    
)



from trl import SFTTrainer, SFTConfig

# Assuming your DatasetDict is named 'reasoning_df' as in your original code
reasoning_formatted = []

# 1. Select the 'train' split (or 'test', 'validation', etc.)

dataset_new= split_dataset['train']
# 2. Iterate directly over the dataset split
for row in dataset_new:
    # 'row' is now a dictionary for each example, e.g., {'problem': '...', 'expected_answer': '...'}
    
    # Construct the user content from the 'messages' column
    user_content = next((item['content'] for item in row['messages'] if item['role'] == 'user'), None)
    
    # Construct the assistant content from the 'messages' column
    assistant_content = next((item['content'] for item in row['messages'] if item['role'] == 'assistant'), None)

    # Make sure both user and assistant content exist before formatting
    if user_content and assistant_content:
        formatted_text = f"""<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
{user_content}
<|im_end|>
<|im_start|>assistant
{assistant_content}
<|im_end|>"""
        reasoning_formatted.append({"text": formatted_text})

# Now reasoning_formatted contains your strings
print(reasoning_formatted[0])


# Create SFT Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
   train_dataset=reasoning_formatted,
    dataset_text_field="text",
    args=training_arguments,
    packing=True,
    max_seq_length=2048
)













trainer = SFTTrainer(
    model=model,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
    peft_config=peft_config,
    max_seq_length=512,
    # REMOVE this line: dataset_text_field="text",
    formatting_func=formatting_func, # ADD this line
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
)








#from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

from transformers import TrainingArguments
training_args = TrainingArguments(
    output_dir="/home/s27mhusa_hpc/Master-Thesis/Fine-tune-LLM-Document/fine_tuned_qwen",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    gradient_checkpointing=False,
    optim="adamw_torch",
    learning_rate=2e-4,
    weight_decay=0.01,
    max_grad_norm=0.3,
    max_steps=30,
    warmup_ratio=0.03,
    logging_steps=1,
    save_strategy="steps",
    save_steps=10,
    save_total_limit=3,
    bf16=False,
    tf32=True,
    lr_scheduler_type="cosine",
    seed=42
)

from transformers import Trainer
trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()