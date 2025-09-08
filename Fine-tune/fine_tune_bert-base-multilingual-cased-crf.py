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


class BertCRFForTokenClassification(nn.Module):
    def __init__(self, model_checkpoint, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.bert = AutoModelForTokenClassification.from_pretrained(
            model_checkpoint, 
            num_labels=num_labels
        )
        # Remove the classifier head from BERT as we'll use CRF
        self.bert.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Filter out training-specific kwargs that BERT doesn't expect
        bert_kwargs = {k: v for k, v in kwargs.items() 
                      if k not in ['labels', 'num_items_in_batch', 'return_loss']}
        
        # Get BERT outputs
        outputs = self.bert.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **bert_kwargs
        )
        
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        
        # Get emission scores from linear layer
        emissions = self.bert.classifier(sequence_output)
        
        if labels is not None:
            # Create mask for CRF (ignore -100 labels)
            mask = (labels != -100) & (attention_mask == 1)
            
            # CRITICAL: Ensure first timestep is True for ALL sequences in training too
            mask[:, 0] = True
            
            # Replace -100 with 0 for CRF computation (will be masked out)
            crf_labels = labels.clone()
            crf_labels[labels == -100] = 0
            
            # Also ensure first token label is not -100 for CRF
            # If first token was -100, set it to O (index 0)
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
            # During inference, decode best path
            mask = attention_mask.bool()
            
            # CRITICAL: Ensure first timestep is True for ALL sequences
            mask[:, 0] = True
            
            print(f"Inference - First timestep mask: {mask[:, 0]}")
            print(f"All first timesteps True: {mask[:, 0].all()}")
            
            try:
                best_paths = self.crf.decode(emissions, mask=mask)
            except Exception as e:
                print(f"CRF Decode Error: {e}")
                print(f"Emissions shape: {emissions.shape}")
                print(f"Mask shape: {mask.shape}")
                print(f"Mask dtype: {mask.dtype}")
                raise
            
            # Convert to tensor format expected by trainer
            predictions = torch.zeros_like(emissions[:, :, 0], dtype=torch.long)
            for i, path in enumerate(best_paths):
                predictions[i, :len(path)] = torch.tensor(path, device=emissions.device)
            
            return TokenClassifierOutput(
                logits=predictions,  # Return decoded predictions as logits
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )


# Tokenization & label alignment
def tokenize_and_align_labels(example):
    tokenized_inputs = tokenizer(
        example["tokens"],
        truncation=True,
        is_split_into_words=True,
        return_offsets_mapping=True,
        padding="max_length",
        max_length=512  # Add explicit max length
    )
    
    labels = []
    word_ids = tokenized_inputs.word_ids()
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            labels.append(-100)
        elif word_idx != previous_word_idx:
            labels.append(example["ner_tags"][word_idx])
        else:
            labels.append(example["ner_tags"][word_idx])  # or -100 to ignore subwords
        previous_word_idx = word_idx

    tokenized_inputs["labels"] = labels
    # Remove offset_mapping as it's not needed for training
    tokenized_inputs.pop("offset_mapping", None)
    return tokenized_inputs


# Modified align predictions for CRF
def align_predictions(predictions, label_ids):
    # Convert predictions to numpy if they're tensors
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(label_ids, torch.Tensor):
        label_ids = label_ids.cpu().numpy()
    
    # If predictions come from CRF decode, they're already argmax'd
    if predictions.ndim == 2:
        preds = predictions
    else:
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


# Custom Trainer to handle CRF predictions
class CRFTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():
            # Set model to eval mode
            model.eval()
            outputs = model(**inputs)
            
            if hasattr(model, 'crf'):
                # For CRF models, get the best path predictions
                emissions = outputs.logits
                attention_mask = inputs.get('attention_mask')
                mask = attention_mask == 1
                
                # Ensure first timestep is always True
                mask[:, 0] = True
                
                predictions = model.crf.decode(emissions, mask=mask)
                
                # Convert to tensor format
                batch_size, max_len = emissions.shape[:2]
                pred_tensor = torch.zeros((batch_size, max_len), dtype=torch.long, device=emissions.device)
                for i, path in enumerate(predictions):
                    pred_tensor[i, :len(path)] = torch.tensor(path, device=emissions.device)
                
                loss = outputs.loss if 'labels' in inputs else None
                
                return (loss, pred_tensor, inputs.get('labels'))
            else:
                return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)


# Compute metrics with seqeval
def compute_metrics(p):
    predictions, label_ids = p
    preds, labels = align_predictions(predictions, label_ids)
    results = seqeval.compute(predictions=preds, references=labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


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

    model_checkpoint = "bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # Apply tokenization
    tokenized_dataset = dataset.map(tokenize_and_align_labels, remove_columns=dataset["train"].column_names)

    # Load metric
    seqeval = evaluate.load("seqeval")

    # Optuna hyperparameter search space
    def optuna_hp_space(trial):
        return {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
                "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 30),
                "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
                "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
            }

    # Modified model init for CRF
    def model_init():
        return BertCRFForTokenClassification(
            model_checkpoint=model_checkpoint,
            num_labels=len(label_list)
        )

    # Static training args
    training_args = TrainingArguments(
        output_dir="/lustre/scratch/data/s27mhusa_hpc-murtuza_master_thesis/ner_crf_model_6thSeptember",
        eval_strategy="epoch",
        save_strategy="epoch", 
        logging_dir="./logs",
        run_name="bert-base-multilingual-cased_crf_optuna_tuning_6thSeptember",
        metric_for_best_model="f1",
        greater_is_better=True,
        load_best_model_at_end=True,
        report_to="wandb",  # Add wandb reporting
        per_device_eval_batch_size=16,  # Add eval batch size
    )

    # Use custom CRF trainer
    trainer = CRFTrainer(
        model_init=model_init,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Run Optuna search
    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        hp_space=optuna_hp_space,
        backend="optuna",
        n_trials=10
    )

    print("Best trial:", best_trial)

    # Retrain with best config
    best_args = TrainingArguments(
        output_dir="/lustre/scratch/data/s27mhusa_hpc-murtuza_master_thesis/ner_crf_model_best_6thSeptember",
        eval_strategy="epoch",
        save_strategy="epoch", 
        logging_dir="./logs_best",
        run_name="bert-base-multilingual-cased_crf_best_run_6thSeptember",
        learning_rate=best_trial.hyperparameters["learning_rate"],
        num_train_epochs=best_trial.hyperparameters["num_train_epochs"],
        per_device_train_batch_size=best_trial.hyperparameters["per_device_train_batch_size"],
        per_device_eval_batch_size=16,
        weight_decay=best_trial.hyperparameters["weight_decay"],
        metric_for_best_model="f1",
        greater_is_better=True,
        load_best_model_at_end=True,
        report_to="wandb",
    )

    # Create final trainer
    final_trainer = CRFTrainer(
        model_init=model_init,
        args=best_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    final_trainer.train()
    
    # Validation predictions
    outputs = final_trainer.predict(tokenized_dataset["validation"])
    preds, labels = align_predictions(outputs.predictions, outputs.label_ids)

    for i in range(min(3, len(preds))):  # Prevent index errors
        print("Pred:", preds[i])
        print("Gold:", labels[i])
        print()

    # Final test evaluation
    results = final_trainer.predict(tokenized_dataset["test"])
    print("Test Results:", results.metrics)