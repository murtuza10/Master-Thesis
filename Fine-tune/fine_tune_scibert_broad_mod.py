import os
import gc
import numpy as np
import torch
import evaluate
import optuna
from datasets import Dataset, DatasetDict, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from seqeval.metrics import classification_report as seqeval_classification_report

# --- Environment Setup and GPU Cache Clearing ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()
gc.collect()

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


# --- Helper Functions (Tokenization, Metrics) ---
# It's assumed that `label_list` and `tokenizer` will be in the global scope
# when these are called.

def tokenize_and_align_labels(examples):
    """Tokenizes texts and aligns labels with sub-word units."""
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
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics(p):
    """Computes and returns a dictionary of performance metrics."""
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
    
# --- 1. SETUP: DATA LOADING AND CONFIGURATION ---

# Load datasets
train_dataset = Dataset.from_json("/home/s27mhusa_hpc/Master-Thesis/Dataset1stSeptember/NER_dataset_sentence_train_stratified.json")
val_dataset = Dataset.from_json("/home/s27mhusa_hpc/Master-Thesis/Dataset1stSeptember/NER_dataset_sentence_val_stratified.json")
test_dataset = Dataset.from_json("/home/s27mhusa_hpc/Master-Thesis/Dataset1stSeptember/Test_NER_dataset.json")

# Combine for k-fold and final training
combined_dataset = concatenate_datasets([train_dataset, val_dataset])

dataset_dict = DatasetDict({
    'train_val': combined_dataset,
    'validation': val_dataset,  # Keep original validation set for monitoring
    'test': test_dataset
})

# Global configurations
label_list = ["O", "B-startTime", "I-startTime", "B-endTime", "I-endTime", "B-city", "I-city", "B-duration", "I-duration", "B-cropSpecies", "I-cropSpecies", "B-region", "I-region", "B-country", "I-country", "B-Soil", "I-Soil"]
model_checkpoint = "allenai/scibert_scivocab_uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
seqeval = evaluate.load("seqeval")

# Tokenize all datasets once
tokenized_datasets = dataset_dict.map(tokenize_and_align_labels, batched=True)

def model_init():
    """Initializes a fresh model for each training run."""
    return AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        num_labels=len(label_list)
    )

# =================================================================================
# PART A: HYPERPARAMETER SEARCH WITH K-FOLD CROSS-VALIDATION
# =================================================================================
print("\n" + "="*80)
print("PART A: STARTING HYPERPARAMETER SEARCH WITH K-FOLD CROSS-VALIDATION")
print("="*80 + "\n")

def objective(trial):
    """The function for Optuna to optimize."""
    # 1. Sample Hyperparameters
    training_args = TrainingArguments(
        output_dir=f"./optuna/trial_{trial.number}",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=1,
        fp16=True,
        # Hyperparameters to tune
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        num_train_epochs=trial.suggest_int("num_train_epochs", 5, 20),
        per_device_train_batch_size=trial.suggest_categorical("per_device_train_batch_size", [2, 4, 8, 16]),
        weight_decay=trial.suggest_float("weight_decay", 1e-3, 0.1, log=True),
    )

    # 2. Perform K-Fold Cross-Validation for the current trial
    n_splits = 3 # Use a smaller k (e.g., 3) for faster hyperparameter search
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_f1_scores = []
    
    for fold, (train_indices, val_indices) in enumerate(kf.split(tokenized_datasets['train_val'])):
        
        fold_train_dataset = tokenized_datasets['train_val'].select(train_indices)
        fold_val_dataset = tokenized_datasets['train_val'].select(val_indices)

        trainer = Trainer(
            model_init=model_init,
            args=training_args,
            train_dataset=fold_train_dataset,
            eval_dataset=fold_val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        trainer.train()
        eval_metrics = trainer.evaluate()
        fold_f1_scores.append(eval_metrics['eval_f1'])

        # Clean up
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

    # 3. Return the average F1 score as the objective to maximize
    average_f1 = np.mean(fold_f1_scores)
    return average_f1


# Create and run the Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10) # n_trials is how many hyperparameter sets to test

print("\nHyperparameter search complete!")
print("Best trial:", study.best_trial.params)
print("Best F1 score (average over folds):", study.best_value)


# =================================================================================
# PART B: TRAIN FINAL MODEL WITH BEST HYPERPARAMETERS
# =================================================================================
print("\n" + "="*80)
print("PART B: TRAINING FINAL MODEL ON ALL DATA WITH BEST HYPERPARAMETERS")
print("="*80 + "\n")

best_params = study.best_trial.params

final_training_args = TrainingArguments(
    output_dir="./results/final_model",
    run_name="scibert-final-training",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_total_limit=1,
    fp16=True,
    # Use best parameters found by Optuna
    learning_rate=best_params['learning_rate'],
    num_train_epochs=best_params['num_train_epochs'],
    per_device_train_batch_size=best_params['per_device_train_batch_size'],
    weight_decay=best_params['weight_decay'],
)

# Use the full train_val set for training and the original validation set for monitoring
final_trainer = Trainer(
    model_init=model_init,
    args=final_training_args,
    train_dataset=tokenized_datasets['train_val'],
    eval_dataset=tokenized_datasets['validation'],  # Now this key exists!
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

final_trainer.train()

# =================================================================================
# PART C: FINAL EVALUATION ON THE UNSEEN TEST SET
# =================================================================================
print("\n" + "="*80)
print("PART C: EVALUATING FINAL MODEL ON THE TEST SET")
print("="*80 + "\n")

test_results = final_trainer.predict(tokenized_datasets['test'])

print("Final Test Set Metrics:")
print(test_results.metrics)

# Save the final model and tokenizer
final_trainer.save_model("./results/final_model_saved")
tokenizer.save_pretrained("./results/final_model_saved")

print_classification_reports(test_results.predictions, test_results.label_ids, "Final Test")

print("\n" + "="*80)
print("WORKFLOW COMPLETED SUCCESSFULLY!")
print("Final model saved to ./results/final_model_saved")
print("="*80)