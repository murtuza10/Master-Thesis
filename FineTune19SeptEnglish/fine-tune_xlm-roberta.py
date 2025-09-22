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
import random
from seqeval.metrics.sequence_labeling import get_entities


# --- Environment Setup and GPU Cache Clearing ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()
gc.collect()

def debug_prediction_alignment(predictions, label_ids, dataset_name="Predictions"):
    """Debug what happens during prediction alignment"""
    print(f"\n=== {dataset_name} Alignment Debug ===")
    
    # Before alignment - raw predictions
    raw_preds = np.argmax(predictions, axis=2)
    print(f"Raw prediction shape: {raw_preds.shape}")
    
    # After alignment
    preds, labels = align_predictions(predictions, label_ids)
    
    # Count entities in aligned predictions
    pred_b_tags = sum(1 for seq in preds for label in seq if label.startswith('B-'))
    pred_seqeval = sum(len(get_entities(seq)) for seq in preds)
    
    # Count entities in aligned true labels  
    true_b_tags = sum(1 for seq in labels for label in seq if label.startswith('B-'))
    true_seqeval = sum(len(get_entities(seq)) for seq in labels)
    
    print(f"\nAfter alignment:")
    print(f"True labels - B- tags: {true_b_tags}, Seqeval entities: {true_seqeval}")
    print(f"Predictions - B- tags: {pred_b_tags}, Seqeval entities: {pred_seqeval}")
    
    # Check if alignment corrupted the true labels
    if true_seqeval < 271:
        print(f"⚠️  Alignment corrupted true labels! Lost {271 - true_seqeval} entities")
    
    # Check model prediction quality
    if pred_seqeval < pred_b_tags:
        print(f"⚠️  Model predictions have invalid BIO sequences! {pred_b_tags - pred_seqeval} rejected")
        
        # Show some examples
        print(f"\nExamples of invalid predictions:")
        shown = 0
        for i, pred_seq in enumerate(preds[:50]):
            b_count = sum(1 for label in pred_seq if label.startswith('B-'))
            seq_count = len(get_entities(pred_seq))
            if b_count > seq_count and shown < 3:
                print(f"Sequence {i}: {pred_seq}")
                print(f"B- tags: {b_count}, Valid entities: {seq_count}")
                shown += 1



def align_predictions(predictions, label_ids):
    preds = np.argmax(predictions, axis=2)
    batch_size, seq_len = preds.shape

    out_label_list = [[] for _ in range(batch_size)]
    out_pred_list = [[] for _ in range(batch_size)]

    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i][j] != -100:
                out_label_list[i].append(label_list[label_ids[i][j]])
                out_pred_list[i].append(label_list[preds[i][j]])

    return out_pred_list, out_label_list

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



label_list = ["O", "B-startTime", "I-startTime", "B-endTime", "I-endTime", "B-city", "I-city", "B-duration", "I-duration", "B-cropSpecies", "I-cropSpecies", "B-region", "I-region", "B-country", "I-country", "B-Soil", "I-Soil"]


# --- Data Augmentation Functions ---
def augment_sequence(tokens, labels, augmentation_prob=0.3):
    """Apply data augmentation techniques to a single sequence"""
    augmented_tokens = tokens.copy()
    augmented_labels = labels.copy()
    
    # 1. Random token masking (only for O labels to avoid corrupting entities)
    if random.random() < augmentation_prob:
        o_indices = [i for i, label in enumerate(labels) if label == 0]  # 0 is "O" label
        if o_indices:
            mask_idx = random.choice(o_indices)
            augmented_tokens[mask_idx] = "[MASK]"
    
    # 2. Random deletion (only for O tokens)
    if random.random() < augmentation_prob * 0.5:  # Lower probability
        o_indices = [i for i, label in enumerate(labels) if label == 0]
        if len(o_indices) > 1:  # Ensure we don't delete all O tokens
            del_idx = random.choice(o_indices)
            augmented_tokens.pop(del_idx)
            augmented_labels.pop(del_idx)
    
    return augmented_tokens, augmented_labels


# Build pools of same-type entity mentions from train set for replacement
from collections import defaultdict
def build_entity_pools(dataset, tokens_key="tokens", tags_key="ner_tags", id2label=None):
    pools = defaultdict(list)
    for ex in dataset:
        toks, tags = ex[tokens_key], ex[tags_key]
        i = 0
        while i < len(toks):
            tag = id2label[tags[i]]
            if tag.startswith("B-"):
                ent_type = tag[2:]
                j = i + 1
                while j < len(toks) and id2label[tags[j]].startswith("I-"):
                    j += 1
                mention = toks[i:j]
                pools[ent_type].append(mention)
                i = j
            else:
                i += 1
    return pools



import random
def augment_with_mention_replacement(tokens, labels, replace_prob=0.2):
    new_toks, new_labels = [], []
    i = 0
    while i < len(tokens):
        tag = id2label[labels[i]]
        if tag.startswith("B-") and random.random() < replace_prob:
            ent_type = tag[2:]
            # locate span
            j = i + 1
            while j < len(tokens) and id2label[labels[j]].startswith("I-"):
                j += 1
            if entity_pools[ent_type]:
                repl = random.choice(entity_pools[ent_type])
                # write replaced tokens and labels
                new_toks.extend(repl)
                new_labels.extend([labels[i]] + [labels[i]+1]*(len(repl)-1))  # assumes I-tag = B-index+1
            else:
                new_toks.extend(tokens[i:j]); new_labels.extend(labels[i:j])
            i = j
        else:
            new_toks.append(tokens[i]); new_labels.append(labels[i]); i += 1
    return new_toks, new_labels

def augment_dataset_controlled(dataset, augmentation_ratio=0.5):
    augmented = []
    for _ in range(int(len(dataset)*augmentation_ratio)):
        ex = dataset[random.randrange(len(dataset))]
        toks, tags = ex["tokens"], ex["ner_tags"]
        toks2, tags2 = augment_with_mention_replacement(toks, tags, replace_prob=0.2)
        augmented.append({"tokens": toks2, "ner_tags": tags2})
    return concatenate_datasets([dataset, Dataset.from_list(augmented)])    



def augment_dataset(dataset, augmentation_ratio=0.2):
    """Augment dataset by creating additional examples"""
    augmented_examples = []
    
    for i in range(int(len(dataset) * augmentation_ratio)):
        # Select random example to augment
        idx = random.randint(0, len(dataset) - 1)
        original_tokens = dataset[idx]["tokens"]
        original_labels = dataset[idx]["ner_tags"]
        
        aug_tokens, aug_labels = augment_sequence(original_tokens, original_labels)
        
        augmented_examples.append({
            "tokens": aug_tokens,
            "ner_tags": aug_labels
        })
    
    # Combine original and augmented data
    augmented_dataset = Dataset.from_list(augmented_examples)
    return concatenate_datasets([dataset, augmented_dataset])

# --- Helper Functions (Tokenization, Metrics) ---
def tokenize_and_align_labels(examples):
    """Tokenizes texts and aligns labels with sub-word units."""
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=256,
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
train_dataset = Dataset.from_json("/home/s27mhusa_hpc/Master-Thesis/Dataset19September/NER_dataset_sentence_English_train_final.json")
val_dataset   = Dataset.from_json("/home/s27mhusa_hpc/Master-Thesis/Dataset19September/NER_dataset_sentence_English_val_final.json")
test_dataset  = Dataset.from_json("/home/s27mhusa_hpc/Master-Thesis/Dataset19September/Test_NER_dataset_English.json")


label_list = ["O", "B-startTime", "I-startTime", "B-endTime", "I-endTime", "B-city", "I-city", "B-duration", "I-duration", "B-cropSpecies", "I-cropSpecies", "B-region", "I-region", "B-country", "I-country", "B-Soil", "I-Soil"]

id2label = {i: l for i, l in enumerate(label_list)}
entity_pools = build_entity_pools(train_dataset, id2label=id2label)


combined_dataset = concatenate_datasets([train_dataset, val_dataset])

# Apply data augmentation to training data
print("Applying data augmentation...")
augmented_train_dataset = augment_dataset_controlled(combined_dataset, augmentation_ratio=3.0)
print(f"Original train size: {len(combined_dataset)}, Augmented size: {len(augmented_train_dataset)}")

# Combine for k-fold and final training

dataset_dict = DatasetDict({
    'train_val': augmented_train_dataset,
    'validation': val_dataset,
    'test': test_dataset
})

# Global configurations
model_checkpoint = "xlm-roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
seqeval = evaluate.load("seqeval")

# Tokenize all datasets once
tokenized_datasets = dataset_dict.map(tokenize_and_align_labels)

def model_init_with_regularization(dropout=0.3):
    """Initializes a model with enhanced regularization."""
    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        num_labels=len(label_list),
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
    )
    return model

# =================================================================================
# PART A: IMPROVED HYPERPARAMETER SEARCH WITH REGULARIZATION FOCUS
# =================================================================================
print("\n" + "="*80)
print("PART A: STARTING IMPROVED HYPERPARAMETER SEARCH")
print("="*80 + "\n")

def objective(trial):
    """Enhanced objective function with regularization focus."""
    # Regularization parameters
    dropout = trial.suggest_float("dropout", 0.2, 0.5)  # Higher dropout range
    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.3, log=True)  # Higher weight decay
    
    # Learning parameters - favor lower learning rates
    learning_rate = trial.suggest_float("learning_rate", 5e-6, 1e-5, log=True)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.1, 0.5)  # Use ratio instead of steps
    
    def model_init_trial():
        return model_init_with_regularization(dropout)
    
    training_args = TrainingArguments(
        output_dir=f"/lustre/scratch/data/s27mhusa_hpc-murtuza_master_thesis/roberta-english-results-broad_3.0-21sept/hyperparameter_search_regularized/trial_{trial.number}",
        eval_strategy="epoch",  # Use epoch-based evaluation for compatibility
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=1,
        fp16=True,
        # Regularization-focused hyperparameters
        learning_rate=learning_rate,
        num_train_epochs=trial.suggest_int("num_train_epochs", 3, 15),  # Fewer epochs
        per_device_train_batch_size=trial.suggest_categorical("per_device_train_batch_size", [2, 4, 8, 16]),
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        # Additional regularization
        max_grad_norm=0.5,  # Gradient clipping
        dataloader_drop_last=True,
        # Logging
        logging_steps=50,
        report_to=None,  # Disable wandb/tensorboard
    )

    # K-Fold Cross-Validation
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_f1_scores = []
    
    for fold, (train_indices, val_indices) in enumerate(kf.split(tokenized_datasets['train_val'])):
        fold_train_dataset = tokenized_datasets['train_val'].select(train_indices)
        fold_val_dataset = tokenized_datasets['train_val'].select(val_indices)

        trainer = Trainer(
            model_init=model_init_trial,
            args=training_args,
            train_dataset=fold_train_dataset,
            eval_dataset=fold_val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )
        
        trainer.train()
        eval_metrics = trainer.evaluate()
        fold_f1_scores.append(eval_metrics['eval_f1'])

        # Clean up
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

    # Return average F1 score
    average_f1 = np.mean(fold_f1_scores)
    return average_f1

# Run Optuna study with more trials for better optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

print("\nHyperparameter search complete!")
print("Best trial:", study.best_trial.params)
print("Best F1 score (average over folds):", study.best_value)

# =================================================================================
# PART B: FINAL MODEL TRAINING WITH BEST PARAMETERS
# =================================================================================
print("\n" + "="*80)
print("PART B: TRAINING FINAL MODEL WITH REGULARIZATION")
print("="*80 + "\n")

best_params = study.best_trial.params

final_training_args = TrainingArguments(
    output_dir="/lustre/scratch/data/s27mhusa_hpc-murtuza_master_thesis/roberta-english-results-broad_no_aug/results/final_model_regularized_3.0-21sept",
    run_name="roberta-english-regularized-training",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_total_limit=2,
    fp16=True,
    # Best parameters from optimization
    learning_rate=best_params['learning_rate'],
    num_train_epochs=best_params['num_train_epochs'],
    per_device_train_batch_size=best_params['per_device_train_batch_size'],
    weight_decay=best_params['weight_decay'],
    warmup_ratio=best_params['warmup_ratio'],
    # Additional regularization
    max_grad_norm=1.0,
    dataloader_drop_last=True,
    # Logging
    logging_steps=100,
    logging_dir="./logs",
    report_to=None,
)

def final_model_init():
    return model_init_with_regularization(dropout=best_params['dropout'])

# Use regular trainer
final_trainer = Trainer(
    model_init=final_model_init,
    args=final_training_args,
    train_dataset=tokenized_datasets['train_val'],
    eval_dataset=tokenized_datasets['validation'],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

final_trainer.train()

# =================================================================================
# PART C: FINAL EVALUATION ON TEST SET
# =================================================================================
print("\n" + "="*80)
print("PART C: EVALUATING FINAL REGULARIZED MODEL")
print("="*80 + "\n")

test_results = final_trainer.predict(tokenized_datasets['test'])

debug_prediction_alignment(test_results.predictions, test_results.label_ids, "Test Predictions")



print("Final Test Set Metrics:")
print(test_results.metrics)

# Save the final model and tokenizer
final_trainer.save_model("/lustre/scratch/data/s27mhusa_hpc-murtuza_master_thesis/roberta-english_final_model_regularized_saved_broad_3.0-21sept")
tokenizer.save_pretrained("/lustre/scratch/data/s27mhusa_hpc-murtuza_master_thesis/roberta-english_final_model_regularized_saved_broad_3.0-21sept")

print_classification_reports(test_results.predictions, test_results.label_ids, "Final Test")

# Additional analysis: Compare validation vs test performance
val_results = final_trainer.predict(tokenized_datasets['validation'])
print("\nValidation Set Metrics for comparison:")
print(val_results.metrics)
print_classification_reports(val_results.predictions, val_results.label_ids, "Validation")

print("\n" + "="*80)
print("REGULARIZED WORKFLOW COMPLETED!")
print("Final model saved to /lustre/scratch/data/s27mhusa_hpc-murtuza_master_thesis/roberta-english_final_model_regularized_saved_broad_3.0-21sept")
print("="*80)