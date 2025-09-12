# -*- coding: utf-8 -*-
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
import torch

# Load dataset
dataset = load_dataset(
    "json", 
    data_files="/home/s27mhusa_hpc/Master-Thesis/Fine-tune-LLM-Document/text2icasa_training_data_1to1_only_method_fertilizer.jsonl",
    split='train'
)
split_dataset = dataset.train_test_split(test_size=2, seed=42)

# Model and tokenizer loading
model_path = "/lustre/scratch/data/s27mhusa_hpc-murtuza_master_thesis/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_4bit=True
)

# PEFT configuration
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

# Formatting function for your data
def formatting_func(example):
    user_content = next((item['content'] for item in example['messages'] if item['role'] == 'user'), None)
    assistant_content = next((item['content'] for item in example['messages'] if item['role'] == 'assistant'), None)
    
    if user_content and assistant_content:
        return f"""<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
{user_content}
<|im_end|>
<|im_start|>assistant
{assistant_content}
<|im_end|>"""
    return ""

# Training arguments
training_args = TrainingArguments(
    output_dir="/home/s27mhusa_hpc/Master-Thesis/Fine-tune-LLM-Document/fine_tuned_llama",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit",
    num_train_epochs=1,
    eval_strategy="steps",
    eval_steps=50,
    logging_steps=10,
    warmup_steps=10,
    learning_rate=2e-4,
    fp16=True,
    save_strategy="steps",
    save_steps=100,
    group_by_length=True,
    dataloader_pin_memory=False
)

# SFT Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
    peft_config=peft_config,
    formatting_func=formatting_func,
    args=training_args
)

# Train the model
trainer.train()
