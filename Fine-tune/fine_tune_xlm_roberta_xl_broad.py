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

    # Load dataset
    train_dataset = Dataset.from_json("/home/s27mhusa_hpc/Master-Thesis/Dataset1stSeptember/NER_dataset_sentence_train_stratified.json")
    val_dataset   = Dataset.from_json("/home/s27mhusa_hpc/Master-Thesis/Dataset1stSeptember/NER_dataset_sentence_val_stratified.json")
    test_dataset  = Dataset.from_json("/home/s27mhusa_hpc/Master-Thesis/Dataset1stSeptember/Test_NER_dataset.json")

    dataset = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })

    # Label mapping
    label_list = ["O", "B-startTime", "I-startTime", "B-endTime", "I-endTime", "B-city", "I-city", "B-duration", "I-duration", "B-cropSpecies", "I-cropSpecies", "B-region", "I-region", "B-country", "I-country", "B-Soil", "I-Soil"]
    label_to_id = {l: i for i, l in enumerate(label_list)}

    model_checkpoint = "facebook/xlm-roberta-xl"
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

    def optuna_hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 30),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [1]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 0.3, log=True),
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.3),
            "optimizer": trial.suggest_categorical("optimizer", ["AdamW", "Adam", "Adafactor"]),
        }

    def model_init():
        return AutoModelForTokenClassification.from_pretrained(
            model_checkpoint,
            num_labels=len(label_list)
        )

    training_args = TrainingArguments(
        output_dir="/lustre/scratch/data/s27mhusa_hpc-murtuza_master_thesis/xlm_roberta_xl_ner_model_3rd_September",
        eval_strategy="epoch",
        save_strategy="epoch", 
        logging_dir="./logs",
        run_name="xlm_roberta_xl_optuna_tuning_3rd_September",
        metric_for_best_model="f1",
        greater_is_better=True,
        load_best_model_at_end=True,
        fp16=True,
        gradient_accumulation_steps=8,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    # Run Optuna search
    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        hp_space=optuna_hp_space,
        backend="optuna",
        n_trials=10
    )

    print("Best trial:", best_trial)
    
    # Optionally retrain with best config:
    best_args = TrainingArguments(
        output_dir="/lustre/scratch/data/s27mhusa_hpc-murtuza_master_thesis/xlm_roberta_xl_ner_model_3rd_September_best",
        eval_strategy="epoch",
        save_strategy="epoch", 
        logging_dir="./logs_best",
        run_name="xlm_roberta_xl_best_run_3rd_September",
        learning_rate=best_trial.hyperparameters["learning_rate"],
        num_train_epochs=best_trial.hyperparameters["num_train_epochs"],
        per_device_train_batch_size=best_trial.hyperparameters["per_device_train_batch_size"],
        weight_decay=best_trial.hyperparameters["weight_decay"],
        metric_for_best_model="f1",
        greater_is_better=True,
        load_best_model_at_end=True,
        fp16=True,
        gradient_accumulation_steps=4,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model_init=model_init,
        args=best_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    
    # Validation evaluation with classification report
    print("\nEvaluating on validation data...")
    val_outputs = trainer.predict(tokenized_dataset["validation"])
    preds, labels = align_predictions(val_outputs.predictions, val_outputs.label_ids)

    # Print sample predictions
    for i in range(3):
        print("Pred:", preds[i])
        print("Gold:", labels[i])
        print()
    
    # Print validation classification report
    print_classification_reports(val_outputs.predictions, val_outputs.label_ids, "Validation")

    # Final test evaluation with classification report
    print("\nEvaluating on test data...")
    test_results = trainer.predict(tokenized_dataset["test"])
    print("\nTest Metrics:")
    print(test_results.metrics)
    
    # Print test classification report
    print_classification_reports(test_results.predictions, test_results.label_ids, "Test")
    
    # Save detailed results to files
    val_preds, val_labels = align_predictions(val_outputs.predictions, val_outputs.label_ids)
    test_preds, test_labels = align_predictions(test_results.predictions, test_results.label_ids)
    
    # Save validation results
    with open("validation_classification_report.txt", "w") as f:
        f.write("VALIDATION CLASSIFICATION REPORTS\n")
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
    
    # Save test results
    with open("test_classification_report.txt", "w") as f:
        f.write("TEST CLASSIFICATION REPORTS\n")
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
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("Classification reports saved to:")
    print("- validation_classification_report.txt")
    print("- test_classification_report.txt")
    print("="*60)
