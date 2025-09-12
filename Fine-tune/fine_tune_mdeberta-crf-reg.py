from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import Dataset, DatasetDict, load_dataset
import evaluate
import numpy as np
import wandb
import optuna
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import TokenClassifierOutput
from torchcrf import CRF  # pip install pytorch-crf
from sklearn.metrics import classification_report
from seqeval.metrics import classification_report as seqeval_classification_report
from sklearn.model_selection import StratifiedKFold
import random
import shutil
import os

def tokenize_and_align_labels(example):
    tokenized_inputs = tokenizer(
        example["tokens"],
        truncation=True,
        is_split_into_words=True,
        return_offsets_mapping=True,
        padding="max_length",
        max_length=512
    )
    
    labels = []
    word_ids = tokenized_inputs.word_ids()
    previous_word_idx = None
    
    for word_idx in word_ids:
        if word_idx is None:
            # Special tokens (CLS, SEP, PAD) get -100
            labels.append(-100)
        elif word_idx != previous_word_idx:
            # First subword of a word gets the label
            labels.append(example["ner_tags"][word_idx])
        else:
            # Subsequent subwords get the same label (or -100 to ignore)
            labels.append(example["ner_tags"][word_idx])
        previous_word_idx = word_idx

    tokenized_inputs["labels"] = labels
    tokenized_inputs.pop("offset_mapping", None)
    return tokenized_inputs

# Data augmentation function
def augment_sentence(tokens, ner_tags, augment_prob=0.3):
    """Simple data augmentation by randomly masking non-entity tokens"""
    if random.random() > augment_prob:
        return tokens, ner_tags
    
    augmented_tokens = tokens.copy()
    augmented_tags = ner_tags.copy()
    
    # Find non-entity positions (O tags)
    non_entity_positions = [i for i, tag in enumerate(ner_tags) if tag == 0]  # 0 is "O"
    
    if non_entity_positions:
        # Randomly mask 10-20% of non-entity tokens
        num_to_mask = max(1, len(non_entity_positions) // 10)
        positions_to_mask = random.sample(non_entity_positions, min(num_to_mask, len(non_entity_positions)))
        
        for pos in positions_to_mask:
            augmented_tokens[pos] = "[MASK]"
    
    return augmented_tokens, augmented_tags

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


class DebertaV3CRFForTokenClassification(nn.Module):
    def __init__(self, model_checkpoint, num_labels, dropout_rate=0.3):
        super().__init__()
        self.num_labels = num_labels
        
        # Load the base mDeBERTa-v3 model
        self.deberta = AutoModelForTokenClassification.from_pretrained(
            model_checkpoint, 
            num_labels=num_labels
        )
        
        # Replace classifier for CRF compatibility with stronger regularization
        hidden_size = self.deberta.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),  # Increased dropout
            nn.Linear(hidden_size, hidden_size // 2),  # Smaller intermediate layer
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_labels)
        )
        
        self.crf = CRF(num_labels, batch_first=True)
        
        # Stronger dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Add layer normalization for better regularization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Filter out training-specific kwargs that might cause issues
        model_kwargs = {k: v for k, v in kwargs.items() 
                       if k not in ['labels', 'num_items_in_batch', 'return_loss', 
                                   'output_attentions', 'output_hidden_states', 'use_cache']}
        
        # Get mDeBERTa-v3 outputs - use the base model
        outputs = self.deberta.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **model_kwargs
        )
        
        sequence_output = outputs.last_hidden_state
        
        # Apply layer normalization and dropout
        sequence_output = self.layer_norm(sequence_output)
        sequence_output = self.dropout(sequence_output)
        
        # Get emission scores from our classifier
        emissions = self.classifier(sequence_output)
        
        if labels is not None:
            # Training mode - compute CRF loss
            mask = (labels != -100) & (attention_mask == 1)
            
            # CRITICAL: Ensure first timestep is True for ALL sequences
            mask[:, 0] = True
            
            # Replace -100 with 0 for CRF computation
            crf_labels = labels.clone()
            crf_labels[labels == -100] = 0
            
            # Ensure first token label is valid for CRF
            crf_labels[:, 0] = torch.where(labels[:, 0] == -100, 
                                         torch.tensor(0, device=labels.device), 
                                         labels[:, 0])
            
            # Compute CRF loss
            log_likelihood = self.crf(emissions, crf_labels, mask=mask, reduction='mean')
            loss = -log_likelihood
            
            return TokenClassifierOutput(
                loss=loss,
                logits=emissions,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        else:
            # Inference mode - decode best path
            mask = attention_mask.bool()
            mask[:, 0] = True  # Ensure first timestep is valid
            
            try:
                best_paths = self.crf.decode(emissions, mask=mask)
            except Exception as e:
                print(f"CRF Decode Error: {e}")
                print(f"Emissions shape: {emissions.shape}")
                print(f"Mask shape: {mask.shape}")
                raise
            
            # Convert paths to tensor format
            predictions = torch.zeros_like(emissions[:, :, 0], dtype=torch.long)
            for i, path in enumerate(best_paths):
                predictions[i, :len(path)] = torch.tensor(path, device=emissions.device)
            
            return TokenClassifierOutput(
                logits=predictions,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )


# FIXED Custom Trainer for mDeBERTa-v3 CRF with regularization
class DebertaCRFTrainer(Trainer):
    def __init__(self, *args, label_smoothing=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_smoothing = label_smoothing
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override compute_loss to add label smoothing and other regularization
        FIXED: Added num_items_in_batch parameter for compatibility with newer transformers
        """
        labels = inputs.get("labels")
        outputs = model(**inputs)
        
        if labels is not None:
            loss = outputs.loss
            
            # Add L2 regularization manually
            l2_lambda = 0.01  # L2 regularization strength
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            loss = loss + l2_lambda * l2_norm
            
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():
            model.eval()
            
            if hasattr(model, 'crf'):
                # Get mDeBERTa outputs
                outputs = model.deberta.deberta(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask']
                )
                
                sequence_output = outputs.last_hidden_state
                sequence_output = model.layer_norm(sequence_output)
                sequence_output = model.dropout(sequence_output)
                emissions = model.classifier(sequence_output)
                
                # Create mask and decode
                attention_mask = inputs.get('attention_mask')
                mask = (attention_mask == 1).bool()
                mask[:, 0] = True  # Ensure first timestep is valid
                
                predictions = model.crf.decode(emissions, mask=mask)
                
                # Convert to tensor format
                batch_size, max_len = emissions.shape[:2]
                pred_tensor = torch.zeros((batch_size, max_len), dtype=torch.long, device=emissions.device)
                
                for i, path in enumerate(predictions):
                    pred_tensor[i, :len(path)] = torch.tensor(path, device=emissions.device)
                
                # Get loss if labels are present
                loss = None
                if 'labels' in inputs and not prediction_loss_only:
                    labels = inputs['labels']
                    mask_loss = (labels != -100) & (attention_mask == 1)
                    mask_loss[:, 0] = True
                    
                    crf_labels = labels.clone()
                    crf_labels[labels == -100] = 0
                    crf_labels[:, 0] = torch.where(labels[:, 0] == -100, 
                                                 torch.tensor(0, device=labels.device), 
                                                 labels[:, 0])
                    
                    log_likelihood = model.crf(emissions, crf_labels, mask=mask_loss, reduction='mean')
                    loss = -log_likelihood
                
                return (loss, pred_tensor, inputs.get('labels'))
            else:
                return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)


# FIXED Cross-validation function
def run_cross_validation(dataset, k_folds=5):
    """Run k-fold cross validation to better assess model performance"""
    # Combine train and validation for CV
    combined_data = []
    for i in range(len(dataset["train"])):
        combined_data.append({
            'tokens': dataset["train"][i]['tokens'],
            'ner_tags': dataset["train"][i]['ner_tags']
        })
    for i in range(len(dataset["validation"])):
        combined_data.append({
            'tokens': dataset["validation"][i]['tokens'],
            'ner_tags': dataset["validation"][i]['ner_tags']
        })
    
    # Simple k-fold (not stratified for simplicity, but you could implement stratification)
    fold_size = len(combined_data) // k_folds
    cv_scores = []
    
    for fold in range(k_folds):
        print(f"\n=== Cross-Validation Fold {fold + 1}/{k_folds} ===")
        
        # Create train/val split for this fold
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size if fold < k_folds - 1 else len(combined_data)
        
        fold_val_data = combined_data[val_start:val_end]
        fold_train_data = combined_data[:val_start] + combined_data[val_end:]
        
        # Convert to datasets
        fold_train_dataset = Dataset.from_list(fold_train_data)
        fold_val_dataset = Dataset.from_list(fold_val_data)
        
        # Apply tokenization
        fold_train_tokenized = fold_train_dataset.map(tokenize_and_align_labels, remove_columns=fold_train_dataset.column_names)
        fold_val_tokenized = fold_val_dataset.map(tokenize_and_align_labels, remove_columns=fold_val_dataset.column_names)
        
        # Train model for this fold - FIXED TRAINING ARGUMENTS
        fold_trainer = DebertaCRFTrainer(
            model_init=lambda: DebertaV3CRFForTokenClassification(
                model_checkpoint=model_checkpoint,
                num_labels=len(label_list),
                dropout_rate=0.4  # Higher dropout for CV
            ),
            args=TrainingArguments(
                output_dir=f"./cv_fold_{fold}",
                eval_strategy="epoch",
                save_strategy="epoch",  # CHANGED: Match eval strategy
                num_train_epochs=10,  # Fewer epochs for CV
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                learning_rate=3e-5,
                weight_decay=0.1,
                warmup_ratio=0.1,
                load_best_model_at_end=True,  # Now this will work
                metric_for_best_model="f1",
                greater_is_better=True,
                logging_steps=50,
                remove_unused_columns=False,
                report_to=None,  # Disable wandb for CV
                save_total_limit=1,  # ADDED: Only keep the best checkpoint
            ),
            train_dataset=fold_train_tokenized,
            eval_dataset=fold_val_tokenized,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        fold_trainer.train()
        fold_results = fold_trainer.evaluate()
        cv_scores.append(fold_results['eval_f1'])
        print(f"Fold {fold + 1} F1 Score: {fold_results['eval_f1']:.4f}")
        
        # ADDED: Clean up fold checkpoint directory to save space
        if os.path.exists(f"./cv_fold_{fold}"):
            shutil.rmtree(f"./cv_fold_{fold}")
    
    mean_cv_score = np.mean(cv_scores)
    std_cv_score = np.std(cv_scores)
    print(f"\n=== Cross-Validation Results ===")
    print(f"Mean F1 Score: {mean_cv_score:.4f} ± {std_cv_score:.4f}")
    print(f"Individual Fold Scores: {[f'{score:.4f}' for score in cv_scores]}")
    
    return mean_cv_score, std_cv_score


if __name__ == "__main__":
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    # Login to wandb
    wandb.login(key="ed7faaa7784428261467aee38c86ccc5c316f954")

    # Load dataset
    train_dataset = Dataset.from_json("/home/s27mhusa_hpc/Master-Thesis/Dataset1stSeptember/NER_dataset_sentence_train_stratified.json")
    val_dataset   = Dataset.from_json("/home/s27mhusa_hpc/Master-Thesis/Dataset1stSeptember/NER_dataset_sentence_val_stratified.json")
    test_dataset  = Dataset.from_json("/home/s27mhusa_hpc/Master-Thesis/Dataset1stSeptember/Test_NER_dataset.json")

    # # Apply data augmentation to training set
    # print("Applying data augmentation...")
    # augmented_train_data = []
    # for example in train_dataset:
    #     # Original example
    #     augmented_train_data.append(example)
        
    #     # Add augmented version with 50% probability
    #     if random.random() < 0.5:
    #         aug_tokens, aug_tags = augment_sentence(example['tokens'], example['ner_tags'])
    #         augmented_train_data.append({
    #             'tokens': aug_tokens,
    #             'ner_tags': aug_tags
    #         })
    
    # # Create augmented training dataset
    # train_dataset = Dataset.from_list(augmented_train_data)
    # print(f"Training set size after augmentation: {len(train_dataset)}")

    dataset = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })

    # Label mapping
    label_list = ["O", "B-startTime", "I-startTime", "B-endTime", "I-endTime", "B-city", "I-city", "B-duration", "I-duration", "B-cropSpecies", "I-cropSpecies", "B-region", "I-region", "B-country", "I-country", "B-Soil", "I-Soil"]
    label_to_id = {l: i for i, l in enumerate(label_list)}

    # Use mDeBERTa-v3-base
    model_checkpoint = "microsoft/mdeberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    # Ensure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply tokenization
    tokenized_dataset = dataset.map(tokenize_and_align_labels, remove_columns=dataset["train"].column_names)

    # Load metric
    seqeval = evaluate.load("seqeval")

    # Run cross-validation first
    print("\n=== Running Cross-Validation ===")
    cv_mean, cv_std = run_cross_validation(dataset, k_folds=3)  # 3-fold due to computational constraints

    # Reduced hyperparameter search space to prevent overfitting
    def optuna_hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),  # Narrower range
            "num_train_epochs": trial.suggest_int("num_train_epochs", 5, 15),  # Fewer epochs
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8]),  # Smaller batches
            "weight_decay": trial.suggest_float("weight_decay", 0.05, 0.3),  # Higher weight decay
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.05, 0.15),
        }

    # Model initialization with stronger regularization
    def model_init():
        return DebertaV3CRFForTokenClassification(
            model_checkpoint=model_checkpoint,
            num_labels=len(label_list),
            dropout_rate=0.4  # Higher dropout
        )

    # Training arguments with stronger regularization
    training_args = TrainingArguments(
        output_dir="/lustre/scratch/data/s27mhusa_hpc-murtuza_master_thesis/mdeberta_v3_ner_crf_model_regularized",
        eval_strategy="epoch",
        save_strategy="epoch", 
        logging_dir="./logs",
        run_name="mdeberta-v3-base_crf_regularized_optuna",
        metric_for_best_model="f1",
        greater_is_better=True,
        load_best_model_at_end=True,
        report_to="wandb",
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,  # More gradient accumulation
        fp16=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        save_total_limit=2,  # Only keep best 2 models
        logging_steps=50,
    )

    # Use custom mDeBERTa CRF trainer with regularization
    trainer = DebertaCRFTrainer(
        model_init=model_init,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # More aggressive early stopping
        label_smoothing=0.1
    )

    # Run Optuna hyperparameter search with fewer trials
    print("\n=== Running Hyperparameter Search ===")
    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        hp_space=optuna_hp_space,
        backend="optuna",
        n_trials=10  # Reduced number of trials
    )

    print("Best trial:", best_trial)

    # Retrain with best configuration and stronger regularization
    best_args = TrainingArguments(
        output_dir="/lustre/scratch/data/s27mhusa_hpc-murtuza_master_thesis/mdeberta_v3_ner_crf_model_best_regularized",
        eval_strategy="epoch",
        save_strategy="epoch", 
        logging_dir="./logs_best",
        run_name="mdeberta-v3-base_crf_best_regularized",
        learning_rate=best_trial.hyperparameters["learning_rate"],
        num_train_epochs=min(best_trial.hyperparameters["num_train_epochs"], 12),  # Cap epochs
        per_device_train_batch_size=best_trial.hyperparameters["per_device_train_batch_size"],
        per_device_eval_batch_size=8,
        weight_decay=max(best_trial.hyperparameters["weight_decay"], 0.1),  # Ensure minimum weight decay
        warmup_ratio=best_trial.hyperparameters.get("warmup_ratio", 0.1),
        metric_for_best_model="f1",
        greater_is_better=True,
        load_best_model_at_end=True,
        report_to="wandb",
        gradient_accumulation_steps=4,
        fp16=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        save_total_limit=1,  # Only keep the best model
        eval_steps=100,
        logging_steps=50,
    )

    # Create final trainer with best hyperparameters and regularization
    final_trainer = DebertaCRFTrainer(
        model_init=model_init,
        args=best_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],  # Earlier stopping
        label_smoothing=0.1
    )

    # Train the final model
    print("\n=== Training Final Model ===")
    final_trainer.train()
    
    # Validation evaluation with classification report
    print("\nEvaluating on validation data...")
    val_outputs = final_trainer.predict(tokenized_dataset["validation"])
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
    test_results = final_trainer.predict(tokenized_dataset["test"])
    print("\nTest Metrics:")
    print(test_results.metrics)
    
    # Print test classification report
    print_classification_reports(test_results.predictions, test_results.label_ids, "Test")
    
    # Compare with cross-validation results
    test_f1 = test_results.metrics.get('test_f1', 0)
    val_f1 = val_outputs.metrics.get('test_f1', 0)
    
    print(f"\n=== PERFORMANCE COMPARISON ===")
    print(f"Cross-Validation F1: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"Final Validation F1: {val_f1:.4f}")
    print(f"Final Test F1: {test_f1:.4f}")
    print(f"Validation-Test Gap: {abs(val_f1 - test_f1):.4f}")
    
    if abs(val_f1 - test_f1) < 0.05:
        print("✅ Good generalization - small validation-test gap!")
    elif abs(val_f1 - test_f1) < 0.10:
        print("⚠️  Moderate overfitting detected")
    else:
        print("❌ Significant overfitting - large validation-test gap")
    
    # Save detailed results to files
    val_preds, val_labels = align_predictions(val_outputs.predictions, val_outputs.label_ids)
    test_preds, test_labels = align_predictions(test_results.predictions, test_results.label_ids)
    
    # Save validation results
    with open("validation_classification_report_regularized.txt", "w") as f:
        f.write("VALIDATION CLASSIFICATION REPORTS (REGULARIZED MODEL)\n")
        f.write("="*60 + "\n\n")
        f.write(f"Cross-Validation F1: {cv_mean:.4f} ± {cv_std:.4f}\n")
        f.write(f"Final Validation F1: {val_f1:.4f}\n\n")
        
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
    with open("test_classification_report_regularized.txt", "w") as f:
        f.write("TEST CLASSIFICATION REPORTS (REGULARIZED MODEL)\n")
        f.write("="*60 + "\n\n")
        f.write(f"Final Test F1: {test_f1:.4f}\n")
        f.write(f"Validation-Test Gap: {abs(val_f1 - test_f1):.4f}\n\n")
        
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
    print("REGULARIZED TRAINING COMPLETED SUCCESSFULLY!")
    print("Classification reports saved to:")
    print("- validation_classification_report_regularized.txt")
    print("- test_classification_report_regularized.txt")
    print("="*60)

    # Save the final model
    final_trainer.save_model("/lustre/scratch/data/s27mhusa_hpc-murtuza_master_thesis/mdeberta_v3_ner_crf_final_model_regularized")
    tokenizer.save_pretrained("/lustre/scratch/data/s27mhusa_hpc-murtuza_master_thesis/mdeberta_v3_ner_crf_final_model_regularized")