import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import Dataset, DatasetDict, load_dataset
import evaluate
import numpy as np
import wandb
import optuna
import torch
import gc
from sklearn.metrics import classification_report
from seqeval.metrics import classification_report as seqeval_classification_report
import random

# Clear GPU cache
torch.cuda.empty_cache()
gc.collect()

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# Enhanced tokenization function
def tokenize_and_align_labels(examples, augment=False):
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
                # Use -100 for subword tokens to avoid duplicating labels
                label_ids.append(-100)
            previous_word_idx = word_idx
        
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Data augmentation function
def augment_dataset(dataset, augmentation_factor=0.2):
    """
    Simple data augmentation by duplicating some samples
    """
    augmented_data = {"tokens": [], "ner_tags": []}
    
    for i, (tokens, tags) in enumerate(zip(dataset["tokens"], dataset["ner_tags"])):
        augmented_data["tokens"].append(tokens)
        augmented_data["ner_tags"].append(tags)
        
        # Randomly augment some samples
        if random.random() < augmentation_factor:
            augmented_data["tokens"].append(tokens)
            augmented_data["ner_tags"].append(tags)
    
    return Dataset.from_dict(augmented_data)

# Helper functions
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

    # Load dataset
    train_dataset = Dataset.from_json("/home/s27mhusa_hpc/Master-Thesis/Dataset1stSeptember/NER_dataset_sentence_train_stratified.json")
    val_dataset   = Dataset.from_json("/home/s27mhusa_hpc/Master-Thesis/Dataset1stSeptember/NER_dataset_sentence_val_stratified.json")
    test_dataset  = Dataset.from_json("/home/s27mhusa_hpc/Master-Thesis/Dataset1stSeptember/Test_NER_dataset.json")

    # # Apply data augmentation to training set
    # print("Applying data augmentation to training set...")
    # train_dataset = augment_dataset(train_dataset, augmentation_factor=0.15)
    # print(f"Training set size after augmentation: {len(train_dataset)}")

    dataset = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
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
        batch_size=50,
        remove_columns=dataset["train"].column_names,
    )

    # Load metric
    seqeval = evaluate.load("seqeval")

    # Simplified hyperparameter space - remove label_smoothing to avoid compatibility issues
    def optuna_hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 5e-6, 5e-5, log=True),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 12),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16]),
            "weight_decay": trial.suggest_float("weight_decay", 0.01, 0.3, log=True),
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.06, 0.2),
        }

    # Enhanced model initialization with regularization
    def model_init():
        try:
            # Try with classifier_dropout first (newer transformers)
            model = AutoModelForTokenClassification.from_pretrained(
                model_checkpoint,
                num_labels=len(label_list),
                hidden_dropout_prob=0.3,
                attention_probs_dropout_prob=0.1,
                classifier_dropout=0.3,
            )
        except TypeError:
            # Fallback for older transformers versions
            model = AutoModelForTokenClassification.from_pretrained(
                model_checkpoint,
                num_labels=len(label_list),
                hidden_dropout_prob=0.3,
                attention_probs_dropout_prob=0.1,
            )
        return model

    # Enhanced training arguments with regularization
    training_args = TrainingArguments(
        output_dir="/lustre/scratch/data/s27mhusa_hpc-murtuza_master_thesis/mdeberta_ner_model_7th_September_ver2_regularized",
        eval_strategy="epoch",
        save_strategy="epoch", 
        logging_dir="./logs",
        logging_steps=50,
        run_name="mdeberta-v3-base_optuna_tuning_7th_September_ver2_regularized",
        metric_for_best_model="f1",
        greater_is_better=True,
        load_best_model_at_end=True,
        fp16=True,
        gradient_accumulation_steps=4,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        # Regularization parameters
        max_grad_norm=1.0,  # Gradient clipping
        lr_scheduler_type="cosine",  # Better learning rate scheduling
        save_total_limit=3,
        eval_steps=100,
        # Remove report_to to avoid wandb integration issues during HP search
        report_to=[],
    )

    # Standard trainer
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)]
    )

    # Run Optuna search with fewer trials
    print("Starting hyperparameter search...")
    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        hp_space=optuna_hp_space,
        backend="optuna",
        n_trials=12  # Reduced for stability
    )

    print("Best trial:", best_trial)
    
    # Train final model with best hyperparameters
    best_args = TrainingArguments(
        output_dir="/lustre/scratch/data/s27mhusa_hpc-murtuza_master_thesis/mdeberta_ner_model_7th_September_best_ver2_regularized",
        eval_strategy="epoch",
        save_strategy="epoch", 
        logging_dir="./logs_best",
        logging_steps=50,
        run_name="mdeberta-v3-base_best_run_7th_September_ver2_regularized",
        learning_rate=best_trial.hyperparameters["learning_rate"],
        num_train_epochs=min(best_trial.hyperparameters["num_train_epochs"], 10),
        per_device_train_batch_size=best_trial.hyperparameters["per_device_train_batch_size"],
        weight_decay=best_trial.hyperparameters["weight_decay"],
        warmup_ratio=best_trial.hyperparameters["warmup_ratio"],
        metric_for_best_model="f1",
        greater_is_better=True,
        load_best_model_at_end=True,
        fp16=True,
        gradient_accumulation_steps=4,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        save_total_limit=2,
        eval_steps=100,
        report_to=["wandb"],  # Enable wandb for final training
    )

    # Create final trainer
    final_trainer = Trainer(
        model_init=model_init,
        args=best_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)]
    )

    # Train the final model
    print("Training final model with best hyperparameters...")
    final_trainer.train()
    
    # Monitor training vs validation performance
    print("\n" + "="*60)
    print("ANALYZING POTENTIAL OVERFITTING")
    print("="*60)
    
    # Evaluate on training set (subset for efficiency)
    train_subset_size = min(1000, len(tokenized_dataset["train"]))
    train_subset = tokenized_dataset["train"].select(range(train_subset_size))
    train_results = final_trainer.predict(train_subset)
    
    print(f"\nTraining subset performance ({train_subset_size} samples):")
    train_f1 = train_results.metrics.get('test_f1', 0)
    print(f"F1 Score: {train_f1:.4f}")
    
    # Validation evaluation
    print("\nEvaluating on validation data...")
    val_outputs = final_trainer.predict(tokenized_dataset["validation"])
    val_f1 = val_outputs.metrics.get('test_f1', 0)
    print(f"Validation F1 Score: {val_f1:.4f}")
    
    # Calculate overfitting indicator
    if train_f1 > 0 and val_f1 > 0:
        overfitting_gap = train_f1 - val_f1
        print(f"\nOverfitting Gap (Train F1 - Val F1): {overfitting_gap:.4f}")
        if overfitting_gap > 0.05:
            print("⚠️  WARNING: Potential overfitting detected!")
        else:
            print("✅ Overfitting appears to be under control.")
    
    preds, labels = align_predictions(val_outputs.predictions, val_outputs.label_ids)

    # Print sample predictions
    print("\nSample predictions:")
    for i in range(min(3, len(preds))):
        print(f"Sample {i+1}:")
        print("Pred:", preds[i][:10])  # Show first 10 tokens
        print("Gold:", labels[i][:10])
        print()
    
    # Print validation classification report
    print_classification_reports(val_outputs.predictions, val_outputs.label_ids, "Validation")

    # Final test evaluation
    print("\nEvaluating on test data...")
    test_results = final_trainer.predict(tokenized_dataset["test"])
    print("\nTest Metrics:")
    print(test_results.metrics)
    
    # Print test classification report
    print_classification_reports(test_results.predictions, test_results.label_ids, "Test")
    
    # Save detailed results to files
    val_preds, val_labels = align_predictions(val_outputs.predictions, val_outputs.label_ids)
    test_preds, test_labels = align_predictions(test_results.predictions, test_results.label_ids)
    
    # Save validation results
    with open("validation_classification_report_regularized.txt", "w") as f:
        f.write("VALIDATION CLASSIFICATION REPORTS (WITH REGULARIZATION)\n")
        f.write("="*60 + "\n\n")
        
        # Token-level report
        flat_val_preds = [item for sublist in val_preds for item in sublist]
        flat_val_labels = [item for sublist in val_labels for item in sublist]
        f.write("Token-level Classification Report:\n")
        f.write("-" * 50 + "\n")
        f.write(classification_report(flat_val_labels, flat_val_preds, zero_division=0))
        f.write("\n\nEntity-level Classification Report:\n")
        f.write("-" * 50 + "\n")
        f.write(seqeval_classification_report(val_labels, val_preds, zero_division=0))
        
        # Add overfitting analysis
        f.write(f"\n\nOverfitting Analysis:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Training F1 (subset): {train_f1:.4f}\n")
        f.write(f"Validation F1: {val_f1:.4f}\n")
        f.write(f"Overfitting Gap: {train_f1 - val_f1:.4f}\n")
    
    # Save test results
    with open("test_classification_report_regularized.txt", "w") as f:
        f.write("TEST CLASSIFICATION REPORTS (WITH REGULARIZATION)\n")
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
    
    # Save hyperparameter results
    with open("best_hyperparameters.txt", "w") as f:
        f.write("BEST HYPERPARAMETERS\n")
        f.write("="*30 + "\n")
        for key, value in best_trial.hyperparameters.items():
            f.write(f"{key}: {value}\n")
        f.write(f"\nBest F1 Score: {best_trial.objective_value:.4f}\n")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY WITH REGULARIZATION!")
    print("Files saved:")
    print("- validation_classification_report_regularized.txt")
    print("- test_classification_report_regularized.txt")
    print("- best_hyperparameters.txt")
    print("="*60)