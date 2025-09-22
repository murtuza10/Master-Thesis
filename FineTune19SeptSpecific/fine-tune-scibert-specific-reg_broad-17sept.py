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
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from seqeval.metrics.sequence_labeling import get_entities
from seqeval.scheme import IOB2

def build_llrd_optimizer_and_scheduler(model, base_lr, weight_decay, num_train_steps, warmup_steps, decay=0.9):
    no_decay = ["bias", "LayerNorm.weight"]
    # collect layers in order: embeddings + encoder.layer.X + classifier
    layers = [model.bert.embeddings] + list(model.bert.encoder.layer)
    lr = base_lr
    param_groups = []

    # top layers get base_lr, lower layers decay by factor
    for layer_idx in reversed(range(len(layers))):
        layer = layers[layer_idx]
        layer_lr = lr
        params_decay = [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)]
        params_nodecay = [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)]
        if params_decay:
            param_groups.append({"params": params_decay, "lr": layer_lr, "weight_decay": weight_decay})
        if params_nodecay:
            param_groups.append({"params": params_nodecay, "lr": layer_lr, "weight_decay": 0.0})
        lr *= decay

    # classifier/head
    head = model.classifier if hasattr(model, "classifier") else model.classifier
    head_params = list(head.named_parameters())
    params_decay = [p for n, p in head_params if not any(nd in n for nd in no_decay)]
    params_nodecay = [p for n, p in head_params if any(nd in n for nd in no_decay)]
    if params_decay:
        param_groups.append({"params": params_decay, "lr": base_lr * 1.5, "weight_decay": weight_decay})
    if params_nodecay:
        param_groups.append({"params": params_nodecay, "lr": base_lr * 1.5, "weight_decay": 0.0})

    optimizer = AdamW(param_groups)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_train_steps)
    return optimizer, scheduler
    
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, ignore_index=-100, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: (batch, seq, num_labels), targets: (batch, seq)
        num_labels = logits.size(-1)
        logits = logits.view(-1, num_labels)
        targets = targets.view(-1)

        valid_mask = targets != self.ignore_index
        logits = logits[valid_mask]
        targets = targets[valid_mask]

        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        targets_one_hot = F.one_hot(targets, num_classes=num_labels).float()

        pt = (probs * targets_one_hot).sum(dim=-1)
        log_pt = (log_probs * targets_one_hot).sum(dim=-1)

        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            loss = -alpha_t * ((1 - pt) ** self.gamma) * log_pt
        else:
            loss = -((1 - pt) ** self.gamma) * log_pt

        return loss.mean() if self.reduction == 'mean' else loss.sum()

# 2) Custom Trainer to use focal loss
from transformers import Trainer

class FocalLossTrainer(Trainer):
    def __init__(self, *args, focal_gamma=2.0, focal_alpha=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss = FocalLoss(gamma=focal_gamma, alpha=focal_alpha, ignore_index=-100)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs.pop("labels", None)
        if labels is None:
            labels = inputs.pop("label")  # robustness
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.focal_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss

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

# mappings
label_list = ["O","B-soilReferenceGroup","I-soilReferenceGroup", "B-soilOrganicCarbon", "I-soilOrganicCarbon", "B-soilTexture", "I-soilTexture", "B-startTime", "I-startTime", "B-endTime", "I-endTime", "B-city", "I-city", "B-duration", "I-duration", "B-cropSpecies", "I-cropSpecies", "B-soilAvailableNitrogen", "I-soilAvailableNitrogen", "B-soilDepth", "I-soilDepth", "B-region", "I-region", "B-country", "I-country", "B-longitude", "I-longitude", "B-latitude", "I-latitude", "B-cropVariety", "I-cropVariety", "B-soilPH", "I-soilPH", "B-soilBulkDensity", "I-soilBulkDensity"]

id2label = {i: l for i, l in enumerate(label_list)}
label2id = {l: i for i, l in enumerate(label_list)}
entity_types = sorted({l[2:] for l in label_list if l != "O"})
B_ID = {t: label2id[f"B-{t}"] for t in entity_types}
I_ID = {t: label2id[f"I-{t}"] for t in entity_types}
B2I = {B_ID[t]: I_ID[t] for t in entity_types}  # robust B->I map

from collections import defaultdict

def build_entity_pools(dataset, tokens_key="tokens", tags_key="ner_tags"):
    pools = defaultdict(list)
    for ex in dataset:
        toks, tags = ex[tokens_key], ex[tags_key]
        i = 0
        while i < len(toks):
            lab = id2label[tags[i]]
            if lab.startswith("B-"):
                t = lab[2:]
                j = i + 1
                # strictly require I of the same type
                while j < len(toks) and id2label[tags[j]] == f"I-{t}":
                    j += 1
                pools[t].append(toks[i:j])
                i = j
            else:
                i += 1
    return pools


import random

def augment_with_mention_replacement(tokens, labels, replace_prob=0.15, same_len=True):
    new_toks, new_labels = [], []
    i = 0
    n = len(tokens)
    while i < n:
        lab = id2label[labels[i]]
        if lab.startswith("B-") and random.random() < replace_prob:
            t = lab[2:]
            j = i + 1
            while j < n and id2label[labels[j]] == f"I-{t}":
                j += 1
            span_len = j - i
            pool = entity_pools.get(t, [])
            if not pool:
                # fallback: keep original span
                new_toks.extend(tokens[i:j])
                new_labels.extend(labels[i:j])
                i = j
                continue
            cands = [m for m in pool if (len(m) == span_len)] if same_len else pool
            repl = random.choice(cands) if cands else random.choice(pool)
            # labels: first token = original B-id, rest = mapped I-id for that B
            b_id = labels[i]
            i_id = B2I.get(b_id, b_id)  # safe fallback to B-id if mapping missing
            new_toks.extend(repl)
            new_labels.extend([b_id] + [i_id] * (len(repl) - 1))
            i = j
        else:
            new_toks.append(tokens[i])
            new_labels.append(labels[i])
            i += 1
    return new_toks, new_labels

def augment_dataset_controlled(dataset, augmentation_ratio=0.3, replace_prob=0.15, same_len=True):
    augmented = []
    for _ in range(int(len(dataset) * augmentation_ratio)):
        ex = dataset[random.randrange(len(dataset))]
        toks, tags = ex["tokens"], ex["ner_tags"]
        toks2, tags2 = augment_with_mention_replacement(
            toks, tags, replace_prob=replace_prob, same_len=same_len
        )
        # basic validation
        if len(toks2) == len(tags2) and toks2 and tags2:
            augmented.append({"tokens": toks2, "ner_tags": tags2})
    return concatenate_datasets([dataset, Dataset.from_list(augmented)])




# --- 1. SETUP: DATA LOADING AND CONFIGURATION ---

# Load datasets
train_dataset = Dataset.from_json("/home/s27mhusa_hpc/Master-Thesis/Dataset19September/NER_dataset_sentence_Specific_train_final.json")
val_dataset   = Dataset.from_json("/home/s27mhusa_hpc/Master-Thesis/Dataset19September/NER_dataset_sentence_Specific_val_final.json")
test_dataset  = Dataset.from_json("/home/s27mhusa_hpc/Master-Thesis/Dataset19September/Test_NER_dataset_Specific.json")



label_list = ["O","B-soilReferenceGroup","I-soilReferenceGroup", "B-soilOrganicCarbon", "I-soilOrganicCarbon", "B-soilTexture", "I-soilTexture", "B-startTime", "I-startTime", "B-endTime", "I-endTime", "B-city", "I-city", "B-duration", "I-duration", "B-cropSpecies", "I-cropSpecies", "B-soilAvailableNitrogen", "I-soilAvailableNitrogen", "B-soilDepth", "I-soilDepth", "B-region", "I-region", "B-country", "I-country", "B-longitude", "I-longitude", "B-latitude", "I-latitude", "B-cropVariety", "I-cropVariety", "B-soilPH", "I-soilPH", "B-soilBulkDensity", "I-soilBulkDensity"]

id2label = {i: l for i, l in enumerate(label_list)}
entity_pools = build_entity_pools(train_dataset)


combined_dataset = concatenate_datasets([train_dataset, val_dataset])

# Apply data augmentation to training data
print("Applying data augmentation...")
aug_train = augment_dataset_controlled(combined_dataset, augmentation_ratio=2.0)
print(f"Original train size: {len(combined_dataset)}, Augmented size: {len(aug_train)}")


dataset_dict = DatasetDict({
    'train_val': aug_train,
    'validation': val_dataset,
    'test': test_dataset
})



# Global configurations
model_checkpoint = "allenai/scibert_scivocab_uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
seqeval = evaluate.load("seqeval")

# Tokenize all datasets once
tokenized_datasets = dataset_dict.map(tokenize_and_align_labels, batched=True)

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
    dropout = trial.suggest_float("dropout", 0.1, 0.5)  # Higher dropout range
    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.4, log=True)  # Higher weight decay
    
    # Learning parameters - favor lower learning rates
    learning_rate = trial.suggest_float("learning_rate", 5e-7, 1e-5, log=True)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.1, 0.5)  # Use ratio instead of steps
    
    def model_init_trial():
        return model_init_with_regularization(dropout)
    
    training_args = TrainingArguments(
        output_dir=f"/lustre/scratch/data/s27mhusa_hpc-murtuza_master_thesis/Scibert-results-Specific/hyperparameter_search_regularized/trial_{trial.number}",
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
    n_splits = 10
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
        
        # trainer = FocalLossTrainer(
        #     model_init=model_init_trial,
        #     args=training_args,
        #     train_dataset=fold_train_dataset,
        #     eval_dataset=fold_val_dataset,
        #     compute_metrics=compute_metrics,
        #     callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        #     focal_gamma=1.0,  # tune 1.5–3.0
        # )

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
study.optimize(objective, n_trials=10)  # Reduced trials for faster testing

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
    output_dir="/lustre/scratch/data/s27mhusa_hpc-murtuza_master_thesis/Scibert-results-Specific/results/Scibert_final_model_regularized_14",
    run_name="scibert-regularized-training",
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

num_train_steps = (len(tokenized_datasets['train_val']) // final_training_args.per_device_train_batch_size) * final_training_args.num_train_epochs
warmup_steps = int(final_training_args.warmup_ratio * num_train_steps)

# model = final_model_init()


# class LLRDTrainer(FocalLossTrainer):
#     def create_optimizer_and_scheduler(self, num_training_steps: int):
#         warmup_steps = int(self.args.warmup_ratio * num_training_steps)
#         opt, sch = build_llrd_optimizer_and_scheduler(
#             model=self.model,
#             base_lr=self.args.learning_rate,
#             weight_decay=self.args.weight_decay,
#             num_train_steps=num_training_steps,
#             warmup_steps=warmup_steps,
#             decay=0.9,  # relax from 0.2
#         )
#         self.optimizer = opt
#         self.lr_scheduler = sch

# final_trainer = LLRDTrainer(
#     model_init=final_model_init,
#     args=final_training_args,
#     train_dataset=tokenized_datasets['train_val'],
#     eval_dataset=tokenized_datasets['validation'],
#     compute_metrics=compute_metrics,
#     callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
# )

final_trainer.train()

# =================================================================================
# PART C: FINAL EVALUATION ON TEST SET
# =================================================================================
print("\n" + "="*80)
print("PART C: EVALUATING FINAL REGULARIZED MODEL")
print("="*80 + "\n")



test_results = final_trainer.predict(tokenized_datasets['test'])


def entity_distribution(labels):
    counts = {}
    for seq in labels:
        for _, ent_type, _ in get_entities(seq):
            counts[ent_type] = counts.get(ent_type, 0) + 1
    return counts

# After alignment
_, gold_labels = align_predictions(test_results.predictions, test_results.label_ids)
print("Entity counts in TEST set (according to seqeval parsing):")
print(entity_distribution(gold_labels))

print("Final Test Set Metrics:")
print(test_results.metrics)

# Save the final model and tokenizer
final_trainer.save_model("/lustre/scratch/data/s27mhusa_hpc-murtuza_master_thesis/scibert_final_model_regularized_saved_specific")
tokenizer.save_pretrained("/lustre/scratch/data/s27mhusa_hpc-murtuza_master_thesis/scibert_final_model_regularized_saved_specific")

print_classification_reports(test_results.predictions, test_results.label_ids, "Final Test")

# Additional analysis: Compare validation vs test performance
val_results = final_trainer.predict(tokenized_datasets['validation'])
print("\nValidation Set Metrics for comparison:")
print(val_results.metrics)
print_classification_reports(val_results.predictions, val_results.label_ids, "Validation")

print("\n" + "="*80)
print("REGULARIZED WORKFLOW COMPLETED!")
print("Final model saved to /lustre/scratch/data/s27mhusa_hpc-murtuza_master_thesis/scibert_final_model_regularized_saved_specific")
print("="*80)