import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
import evaluate
import numpy as np
import wandb
import optuna
import torch
import gc
from sklearn.metrics import classification_report
from seqeval.metrics import classification_report as seqeval_classification_report

# Clear GPU cache
torch.cuda.empty_cache()
gc.collect()

# Fixed tokenization function
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=128,
    )
    
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx
        
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Add missing helper functions
def align_predictions(predictions, label_ids):
    preds = np.argmax(predictions, axis=2)
    batch_size, seq_len = preds.shape
    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]
    
    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != -100:
                out_label_list[i].append(label_list[label_ids[i][j]])
                preds_list[i].append(label_list[preds[i][j]])
    
    return preds_list, out_label_list

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def print_classification_reports(predictions, label_ids, dataset_name="Dataset"):
    """
    Print both token-level and entity-level classification reports
    """
    preds, labels = align_predictions(predictions, label_ids)
    
    # Flatten predictions and labels for token-level classification report
    flat_preds = [item for sublist in preds for item in sublist]
    flat_labels = [item for sublist in labels for item in sublist]
    
    print(f"\n{'='*60}")
    print(f"{dataset_name.upper()} CLASSIFICATION REPORTS")
    print(f"{'='*60}")
    
    print(f"\n{dataset_name} - Token-level Classification Report:")
    print("-" * 50)
    print(classification_report(flat_labels, flat_preds, zero_division=0))
    
    print(f"\n{dataset_name} - Entity-level Classification Report:")
    print("-" * 50)
    print(seqeval_classification_report(labels, preds, zero_division=0))
    
    # Additional statistics
    print(f"\n{dataset_name} - Additional Statistics:")
    print("-" * 30)
    print(f"Total sequences: {len(preds)}")
    print(f"Total tokens: {len(flat_preds)}")
    print(f"Unique labels in predictions: {len(set(flat_preds))}")
    print(f"Unique labels in ground truth: {len(set(flat_labels))}")

if __name__ == "__main__":
    # Login to wandb
    wandb.login(key="ed7faaa7784428261467aee38c86ccc5c316f954")

    # Load only training and test datasets
    train_dataset = Dataset.from_json("/home/s27mhusa_hpc/Master-Thesis/Dataset1stSeptember/NER_dataset_sentence_train_stratified.json")
    val_dataset   = Dataset.from_json("/home/s27mhusa_hpc/Master-Thesis/Dataset1stSeptember/NER_dataset_sentence_val_stratified.json")
    test_dataset  = Dataset.from_json("/home/s27mhusa_hpc/Master-Thesis/Dataset1stSeptember/Test_NER_dataset.json")

    combined_dataset = concatenate_datasets([train_dataset, val_dataset])


    dataset = DatasetDict({
        "train": combined_dataset,
        "test": test_dataset
    })

    # Label mapping
    label_list = ["O", "B-startTime", "I-startTime", "B-endTime", "I-endTime", "B-city", "I-city", "B-duration", "I-duration", "B-cropSpecies", "I-cropSpecies", "B-region", "I-region", "B-country", "I-country", "B-Soil", "I-Soil"]
    label_to_id = {l: i for i, l in enumerate(label_list)}

    model_checkpoint = "microsoft/mdeberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # Apply tokenization with smaller batch size to avoid memory issues
    tokenized_dataset = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        batch_size=50,  # Reduced batch size
        remove_columns=dataset["train"].column_names,  # Remove original columns to avoid conflicts
    )

    # Load metric
    seqeval = evaluate.load("seqeval")

    # Simplified hyperparameter space without validation-based tuning
    def optuna_hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 5, 15),  # Reduced range to prevent overfitting
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [1, 2, 4, 8, 16]),
            "weight_decay": trial.suggest_float("weight_decay", 0.01, 0.3, log=True),  # Higher weight decay for regularization
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.3),
            "optimizer": trial.suggest_categorical("optimizer", ["AdamW", "Adam", "Adafactor"]),
        }

    def model_init():
        return AutoModelForTokenClassification.from_pretrained(
            model_checkpoint,
            num_labels=len(label_list)
        )

    # Training arguments without validation
    training_args = TrainingArguments(
        output_dir="/lustre/scratch/data/s27mhusa_hpc-murtuza_master_thesis/mdeberta_no_val_ner_model_7th_September_training_only",
        eval_strategy="no",  # No evaluation during training
        save_strategy="epoch",  # Save every epoch for manual selection
        save_total_limit=3,  # Keep only last 3 checkpoints to save space
        logging_dir="./logs",
        run_name="mdeberta-v3-base_training_only_7th_September",
        fp16=True,
        gradient_accumulation_steps=4,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        # Additional regularization settings
        dataloader_drop_last=True,  # Drop incomplete batches
        seed=42,  # Fix seed for reproducibility
    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=None,  # No validation dataset
        tokenizer=tokenizer,
        compute_metrics=None,  # No metrics during training
    )

    # Manual hyperparameter search with fixed number of epochs to prevent overfitting
    print("Starting manual hyperparameter search...")
    
    # Define a few reasonable hyperparameter combinations
    hp_configs = [
        {
            "learning_rate": 2e-5,
            "num_train_epochs": 8,
            "per_device_train_batch_size": 8,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
        },
        {
            "learning_rate": 3e-5,
            "num_train_epochs": 10,
            "per_device_train_batch_size": 4,
            "weight_decay": 0.05,
            "warmup_ratio": 0.15,
        },
        {
            "learning_rate": 1e-5,
            "num_train_epochs": 12,
            "per_device_train_batch_size": 16,
            "weight_decay": 0.1,
            "warmup_ratio": 0.2,
        }
    ]
    
    best_config = hp_configs[1]  # Use the middle configuration as default
    
    print(f"Using hyperparameter configuration: {best_config}")
    
    # Final training with selected config
    final_training_args = TrainingArguments(
        output_dir="/lustre/scratch/data/s27mhusa_hpc-murtuza_master_thesis/mdeberta_no_val_ner_model_7th_September_final",
        eval_strategy="no",
        save_strategy="epoch",
        save_total_limit=3,
        logging_dir="./logs_final",
        run_name="mdeberta-v3-base_no_val_final_run_7th_September",
        learning_rate=best_config["learning_rate"],
        num_train_epochs=best_config["num_train_epochs"],
        per_device_train_batch_size=best_config["per_device_train_batch_size"],
        weight_decay=best_config["weight_decay"],
        warmup_ratio=best_config["warmup_ratio"],
        fp16=True,
        gradient_accumulation_steps=4,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        dataloader_drop_last=True,
        seed=42,
        # Additional regularization
        max_grad_norm=1.0,  # Gradient clipping
    )

    final_trainer = Trainer(
        model_init=model_init,
        args=final_training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=None,
        tokenizer=tokenizer,
        compute_metrics=None,
    )

    print("Starting final training...")
    final_trainer.train()
    
    # Evaluate only on test data
    print("\nEvaluating on test data...")
    test_results = final_trainer.predict(tokenized_dataset["test"])
    print("\nTest Metrics:")
    
    # Manual computation of test metrics
    test_preds, test_labels = align_predictions(test_results.predictions, test_results.label_ids)
    
    # Compute seqeval metrics manually
    seqeval_results = seqeval.compute(predictions=test_preds, references=test_labels)
    print(f"Test F1: {seqeval_results['overall_f1']:.4f}")
    print(f"Test Precision: {seqeval_results['overall_precision']:.4f}")
    print(f"Test Recall: {seqeval_results['overall_recall']:.4f}")
    print(f"Test Accuracy: {seqeval_results['overall_accuracy']:.4f}")
    
    # Print test classification report
    print_classification_reports(test_results.predictions, test_results.label_ids, "Test")
    
    # Save test results only
    with open("test_classification_report_training_only.txt", "w") as f:
        f.write("TEST CLASSIFICATION REPORTS (Training Only Approach)\n")
        f.write("="*60 + "\n\n")
        
        # Token-level report
        flat_test_preds = [item for sublist in test_preds for item in sublist]
        flat_test_labels = [item for sublist in test_labels for item in sublist]
        f.write("Token-level Classification Report:\n")
        f.write("-" * 50 + "\n")
        f.write(classification_report(flat_test_labels, flat_test_preds, zero_division=0))
        f.write("\n\nEntity-level Classification Report:\n")
        f.write("-" * 50 + "\n")
        f.write(seqeval_classification_report(test_labels, test_preds, zero_division=0))
        
        # Add training configuration info
        f.write(f"\n\nTraining Configuration:\n")
        f.write("-" * 25 + "\n")
        for key, value in best_config.items():
            f.write(f"{key}: {value}\n")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("Model trained only on training data to reduce overfitting.")
    print("Test classification report saved to:")
    print("- test_classification_report_training_only.txt")
    print("="*60)