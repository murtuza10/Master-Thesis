from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import Dataset, DatasetDict, load_dataset
import evaluate
import numpy as np
import wandb
import optuna


# Tokenization & label alignment
def tokenize_and_align_labels(example):
    tokenized_inputs = tokenizer(
        example["tokens"],
        truncation=True,
        is_split_into_words=True,
        return_offsets_mapping=True,
        padding="max_length"
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
    return tokenized_inputs


# Align predictions for metrics
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
    # dataset = load_dataset("json", data_files="/home/s27mhusa_hpc/Master-Thesis/ner_dataset_sentence.json")
    train_dataset = Dataset.from_json("/home/s27mhusa_hpc/Master-Thesis/FinalDatasets-21July/combine_ner_dataset_sentence_train_stratified_filtered.json")
    val_dataset   = Dataset.from_json("/home/s27mhusa_hpc/Master-Thesis/FinalDatasets-21July/combine_ner_dataset_sentence_val_stratified_filtered.json")
    test_dataset  = Dataset.from_json("/home/s27mhusa_hpc/Master-Thesis/FinalDatasets-21July/Test_ner_dataset_sentence.json")

    dataset = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })

    # Label mapping
    label_list = ["O","B-soilReferenceGroup","I-soilReferenceGroup", "B-soilOrganicCarbon", "I-soilOrganicCarbon", "B-soilTexture", "I-soilTexture", "B-startTime", "I-startTime", "B-endTime", "I-endTime", "B-city", "I-city", "B-duration", "I-duration", "B-cropSpecies", "I-cropSpecies", "B-soilAvailableNitrogen", "I-soilAvailableNitrogen", "B-soilDepth", "I-soilDepth", "B-region", "I-region", "B-country", "I-country", "B-longitude", "I-longitude", "B-latitude", "I-latitude", "B-cropVariety", "I-cropVariety", "B-soilPH", "I-soilPH", "B-soilBulkDensity", "I-soilBulkDensity"]
    label_to_id = {l: i for i, l in enumerate(label_list)}

    model_checkpoint = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    tokenized_dataset = dataset.map(tokenize_and_align_labels)

    # Load metric
    seqeval = evaluate.load("seqeval")

    # Optuna hyperparameter search space
    def optuna_hp_space(trial):
        return {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
                "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 30),  # Increased max to 30
                "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
                "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
            }

    # Model init for trainer re-instantiation
    def model_init():
        return AutoModelForTokenClassification.from_pretrained(
            model_checkpoint,
            num_labels=len(label_list)
        )

    # Static training args (some overridden by Optuna)
    training_args = TrainingArguments(
        output_dir="/lustre/scratch/data/s27mhusa_hpc-murtuza_master_thesis/ner_model_22_July",
        eval_strategy="epoch",
        save_strategy="epoch", 
        logging_dir="./logs",
        run_name="bert_base_cased_1_optuna_tuning_22_July",
        metric_for_best_model="f1",
        greater_is_better=True,
        load_best_model_at_end=True
    )

    # Trainer setup for hyperparameter search
    trainer = Trainer(
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
        n_trials=5
    )

    print("Best trial:", best_trial)

    # Optionally retrain with best config:
    best_args = TrainingArguments(
        output_dir="/lustre/scratch/data/s27mhusa_hpc-murtuza_master_thesis/ner_model_best_22_July",
        eval_strategy="epoch",
        save_strategy="epoch", 
        logging_dir="./logs_best",
        run_name="bert_base_cased_1_best_run_22_July",
        learning_rate=best_trial.hyperparameters["learning_rate"],
        num_train_epochs=best_trial.hyperparameters["num_train_epochs"],
        per_device_train_batch_size=best_trial.hyperparameters["per_device_train_batch_size"],
        weight_decay=best_trial.hyperparameters["weight_decay"],
        metric_for_best_model="f1",
        greater_is_better=True,
        load_best_model_at_end=True
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
    outputs = trainer.predict(tokenized_dataset["validation"])
    preds, labels = align_predictions(outputs.predictions, outputs.label_ids)

    for i in range(3):
        print("Pred:", preds[i])
        print("Gold:", labels[i])
        print()

    # Final test evaluation
    results = trainer.predict(tokenized_dataset["test"])
    print(results.metrics)
