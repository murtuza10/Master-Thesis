import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer
from transformers import DataCollatorForTokenClassification
from datasets import Dataset
from sklearn.metrics import classification_report
from seqeval.metrics import classification_report as seqeval_classification_report
from seqeval.metrics import f1_score as seqeval_f1_score
from seqeval.scheme import IOB2
import numpy as np
import json
import os
import sys
from datetime import datetime

# Add the power monitoring module to the path
sys.path.append('/home/s27mhusa_hpc/Master-Thesis/FineTune19SeptBroad')
from power_monitor import power_monitor, estimate_carbon_footprint

# Model and tokenizer paths
model_path = "/lustre/scratch/data/s27mhusa_hpc-murtuza_master_thesis/scibert_final_english_model_regularized_saved_broad-3"

def compute_advanced_f1_scores(predictions, label_ids):
    """Advanced F1 computation with boundary-relaxed evaluation"""
    preds, labels = align_predictions(predictions, label_ids)
    
    # Standard exact matching
    exact_f1 = seqeval_f1_score(labels, preds, mode='strict', scheme=IOB2, average='weighted')
    
    # Custom partial matching implementation
    def partial_match_f1(true_entities, pred_entities):
        """Calculate F1 with partial boundary matching"""
        tp = fp = fn = 0
        
        for true_seq, pred_seq in zip(true_entities, pred_entities):
            true_set = set(extract_entities(true_seq))
            pred_set = set(extract_entities(pred_seq))
            
            # Count exact matches
            exact_matches = true_set & pred_set
            tp += len(exact_matches)
            
            # Count partial matches (overlapping boundaries with same type)
            remaining_true = true_set - exact_matches
            remaining_pred = pred_set - exact_matches
            
            for true_ent in remaining_true:
                for pred_ent in remaining_pred:
                    if has_overlap(true_ent, pred_ent) and same_type(true_ent, pred_ent):
                        tp += 0.5  # Partial credit
                        remaining_pred.remove(pred_ent)
                        break
            
            fp += len(remaining_pred)
            fn += len(remaining_true)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return f1
    
    # Helper functions for entity extraction and overlap detection
    def extract_entities(sequence):
        entities = []
        current_entity = None
        for i, label in enumerate(sequence):
            if label.startswith('B-'):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {'start': i, 'end': i, 'type': label[2:]}
            elif label.startswith('I-') and current_entity and label[2:] == current_entity['type']:
                current_entity['end'] = i
            else:
                if current_entity:
                    entities.append(current_entity)
                current_entity = None
        if current_entity:
            entities.append(current_entity)
        return [(e['start'], e['end'], e['type']) for e in entities]
    
    def has_overlap(ent1, ent2):
        return not (ent1[1] < ent2[0] or ent2[1] < ent1[0])
    
    def same_type(ent1, ent2):
        return ent1[2] == ent2[2]
    
    partial_f1 = partial_match_f1(labels, preds)
    
    return exact_f1, partial_f1, preds, labels


def tokenize_and_align_labels(examples, tokenizer, label2id):
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


def load_saved_model(model_path):
    """Load the saved model and tokenizer"""
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    
    # Force CPU usage for power monitoring
    model.to('cpu')
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Number of labels: {model.config.num_labels}")
    print(f"Model device: {next(model.parameters()).device}")
    
    return model, tokenizer

label_list = ["O", "B-startTime", "I-startTime", "B-endTime", "I-endTime", "B-city", "I-city", "B-duration", "I-duration", "B-cropSpecies", "I-cropSpecies", "B-region", "I-region", "B-country", "I-country", "B-Soil", "I-Soil"]

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

def compute_f1_scores(predictions, label_ids):
    """Compute comprehensive F1 scores for NER evaluation"""
    preds, labels = align_predictions(predictions, label_ids)
    
    # Flatten predictions and labels for token-level evaluation
    flat_preds = [item for sublist in preds for item in sublist]
    flat_labels = [item for sublist in labels for item in sublist]
    
    # 1. Exact F1 Score (Strict Entity Matching)
    # Complete entity boundary and type must match exactly
    exact_f1 = seqeval_f1_score(labels, preds, mode='strict', scheme=IOB2, average='weighted')
    
    # 2. Partial F1 Score (Multiple approaches for robustness)
    try:
        # Try seqeval lenient mode first (if available)
        partial_f1 = seqeval_f1_score(labels, preds, mode='lenient', scheme=IOB2, average='weighted')
    except:
        # Fallback to token-level weighted F1
        from sklearn.metrics import f1_score
        partial_f1 = f1_score(flat_labels, flat_preds, average='weighted', zero_division=0)
    
    # 3. Additional meaningful metrics
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    # Token-level metrics (micro-averaged)
    token_f1 = f1_score(flat_labels, flat_preds, average='micro', zero_division=0)
    token_precision = precision_score(flat_labels, flat_preds, average='micro', zero_division=0)
    token_recall = recall_score(flat_labels, flat_preds, average='micro', zero_division=0)
    
    # Entity-level precision and recall for completeness
    from seqeval.metrics import precision_score as seqeval_precision
    from seqeval.metrics import recall_score as seqeval_recall
    
    entity_precision = seqeval_precision(labels, preds, scheme=IOB2)
    entity_recall = seqeval_recall(labels, preds, scheme=IOB2)
    
    return {
        'exact_f1': exact_f1,           # Strict entity-level F1
        'partial_f1': partial_f1,       # Lenient/token-level F1
        'token_f1': token_f1,           # Token-level micro F1
        'token_precision': token_precision,
        'token_recall': token_recall,
        'entity_precision': entity_precision,
        'entity_recall': entity_recall,
        'predictions': preds,
        'labels': labels
    }


def save_results_to_file(results_dict, output_dir="./evaluation_results"):
    """Save evaluation results to multiple file formats with power monitoring data"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"model_evaluation_with_power_{timestamp}"
    
    # Save detailed results as JSON
    json_file = os.path.join(output_dir, f"{base_filename}.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    
    # Save summary results as text file
    txt_file = os.path.join(output_dir, f"{base_filename}_summary.txt")
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("MODEL EVALUATION RESULTS WITH POWER MONITORING\n")
        f.write("="*80 + "\n")
        f.write(f"Evaluation Date: {results_dict['evaluation_info']['timestamp']}\n")
        f.write(f"Model Path: {results_dict['evaluation_info']['model_path']}\n")
        f.write(f"Test Dataset Size: {results_dict['evaluation_info']['test_dataset_size']}\n")
        f.write(f"Labels Used: {results_dict['evaluation_info']['labels']}\n")
        
        # Add power consumption information
        if 'power_consumption' in results_dict:
            power = results_dict['power_consumption']
            f.write("\n" + "="*80 + "\n")
            f.write("POWER CONSUMPTION ANALYSIS\n")
            f.write("="*80 + "\n")
            f.write(f"Duration: {power['duration_seconds']:.2f} seconds ({power['duration_minutes']:.2f} minutes)\n")
            f.write(f"Total Energy: {power['total_energy_wh']:.4f} Wh ({power['total_energy_kwh']:.6f} kWh)\n")
            f.write(f"Average Power: {power['average_power_watts']:.2f} W\n")
            f.write(f"Peak Power: {power['peak_power_watts']:.2f} W\n")
            f.write(f"Min Power: {power['min_power_watts']:.2f} W\n")
            f.write(f"Average CPU Usage: {power['average_cpu_usage_percent']:.1f}%\n")
            f.write(f"Peak CPU Usage: {power['peak_cpu_usage_percent']:.1f}%\n")
            f.write(f"Average Memory Usage: {power['average_memory_usage_percent']:.1f}%\n")
            f.write(f"Peak Memory Usage: {power['peak_memory_usage_percent']:.1f}%\n")
        
        # Add carbon footprint information
        if 'carbon_footprint' in results_dict:
            carbon = results_dict['carbon_footprint']
            f.write("\n" + "="*80 + "\n")
            f.write("ENVIRONMENTAL IMPACT\n")
            f.write("="*80 + "\n")
            f.write(f"CO2 Emissions: {carbon['co2_g']:.2f} g\n")
            f.write(f"Trees needed to offset: {carbon['trees_needed_for_offset']:.4f}\n")
            f.write(f"Equivalent car travel: {carbon['car_km_equivalent']:.2f} km\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("F1 SCORES\n")
        f.write("="*80 + "\n")
        f.write(f"Exact F1 Score (Strict Entity Matching): {results_dict['f1_scores']['exact_f1']:.4f}\n")
        f.write(f"Partial F1 Score (Lenient/Token-level): {results_dict['f1_scores']['partial_f1']:.4f}\n")
        f.write("\n" + "="*80 + "\n")
        f.write("CLASSIFICATION REPORTS\n")
        f.write("="*80 + "\n")
        f.write("\nTOKEN-LEVEL CLASSIFICATION REPORT:\n")
        f.write("-" * 50 + "\n")
        f.write(results_dict['classification_reports']['token_level'])
        f.write("\n\nENTITY-LEVEL CLASSIFICATION REPORT:\n")
        f.write("-" * 50 + "\n")
        f.write(results_dict['classification_reports']['entity_level'])
        f.write("\n\n" + "="*80 + "\n")
        f.write("ADDITIONAL STATISTICS\n")
        f.write("="*80 + "\n")
        stats = results_dict['statistics']
        f.write(f"Total sequences: {stats['total_sequences']}\n")
        f.write(f"Total tokens: {stats['total_tokens']}\n")
        f.write(f"Unique labels in predictions: {stats['unique_labels_pred']}\n")
        f.write(f"Unique labels in ground truth: {stats['unique_labels_true']}\n")
    
    # Save predictions as JSON for further analysis
    predictions_file = os.path.join(output_dir, f"{base_filename}_predictions.json")
    predictions_data = {
        'predictions': results_dict['predictions'],
        'ground_truth': results_dict['ground_truth'],
        'evaluation_info': results_dict['evaluation_info']
    }
    with open(predictions_file, 'w', encoding='utf-8') as f:
        json.dump(predictions_data, f, indent=2, ensure_ascii=False)
    
    # Create a simple CSV with key metrics for easy reference
    csv_file = os.path.join(output_dir, f"{base_filename}_metrics.csv")
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write("metric,value\n")
        f.write(f"exact_f1,{results_dict['f1_scores']['exact_f1']:.4f}\n")
        f.write(f"partial_f1,{results_dict['f1_scores']['partial_f1']:.4f}\n")
        f.write(f"total_sequences,{results_dict['statistics']['total_sequences']}\n")
        f.write(f"total_tokens,{results_dict['statistics']['total_tokens']}\n")
        f.write(f"test_dataset_size,{results_dict['evaluation_info']['test_dataset_size']}\n")
        
        # Add power metrics to CSV
        if 'power_consumption' in results_dict:
            power = results_dict['power_consumption']
            f.write(f"total_energy_wh,{power['total_energy_wh']:.4f}\n")
            f.write(f"average_power_watts,{power['average_power_watts']:.2f}\n")
            f.write(f"peak_power_watts,{power['peak_power_watts']:.2f}\n")
            f.write(f"duration_seconds,{power['duration_seconds']:.2f}\n")
        
        if 'carbon_footprint' in results_dict:
            carbon = results_dict['carbon_footprint']
            f.write(f"co2_emissions_g,{carbon['co2_g']:.2f}\n")
    
    print(f"\n📁 Results saved to:")
    print(f"   📄 Summary: {txt_file}")
    print(f"   📊 Detailed JSON: {json_file}")
    print(f"   🎯 Predictions: {predictions_file}")
    print(f"   📈 CSV Metrics: {csv_file}")
    
    return {
        'summary_file': txt_file,
        'json_file': json_file,
        'predictions_file': predictions_file,
        'csv_file': csv_file
    }

def print_classification_reports(predictions, label_ids, dataset_name="Dataset", save_reports=False):
    """Print both token-level and entity-level classification reports"""
    preds, labels = align_predictions(predictions, label_ids)
    
    # Flatten predictions and labels for token-level classification report
    flat_preds = [item for sublist in preds for item in sublist]
    flat_labels = [item for sublist in labels for item in sublist]
    
    print(f"\n{'='*60}")
    print(f"{dataset_name.upper()} CLASSIFICATION REPORTS")
    print(f"{'='*60}")
    
    print(f"\n{dataset_name} - Token-level Classification Report:")
    print("-" * 50)
    token_report = classification_report(flat_labels, flat_preds, zero_division=0)
    print(token_report)
    
    print(f"\n{dataset_name} - Entity-level Classification Report:")
    print("-" * 50)
    entity_report = seqeval_classification_report(labels, preds, zero_division=0)
    print(entity_report)
    
    # Additional statistics
    print(f"\n{dataset_name} - Additional Statistics:")
    print("-" * 30)
    stats = {
        'total_sequences': len(preds),
        'total_tokens': len(flat_preds),
        'unique_labels_pred': len(set(flat_preds)),
        'unique_labels_true': len(set(flat_labels))
    }
    
    print(f"Total sequences: {stats['total_sequences']}")
    print(f"Total tokens: {stats['total_tokens']}")
    print(f"Unique labels in predictions: {stats['unique_labels_pred']}")
    print(f"Unique labels in ground truth: {stats['unique_labels_true']}")
    
    if save_reports:
        return token_report, entity_report, stats, preds, labels
    else:
        return preds, labels

def evaluate_model_on_test_set_with_power_monitoring(model, tokenizer, test_dataset, compute_metrics_func=None, monitoring_interval=0.1):
    """Evaluate the model on test dataset with power monitoring"""
    
    # Create data collator for token classification
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
        max_length=512,
        pad_to_multiple_of=None,
        return_tensors="pt",
    )
    
    # Create trainer for evaluation
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_func,
    )
    
    print("Running evaluation on test dataset with power monitoring...")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Power monitoring interval: {monitoring_interval}s")
    
    # Start power monitoring and run evaluation
    with power_monitor(monitoring_interval) as monitor:
        # Get predictions
        predictions = trainer.predict(test_dataset)
    
    # Get power monitoring results
    power_results = monitor.get_results()
    
    # Print power consumption summary
    print("\n" + "="*80)
    print("POWER CONSUMPTION SUMMARY")
    print("="*80)
    print(f"Duration: {power_results['duration_seconds']:.2f} seconds ({power_results['duration_minutes']:.2f} minutes)")
    print(f"Total Energy: {power_results['total_energy_wh']:.4f} Wh")
    print(f"Average Power: {power_results['average_power_watts']:.2f} W")
    print(f"Peak Power: {power_results['peak_power_watts']:.2f} W")
    print(f"Average CPU Usage: {power_results['average_cpu_usage_percent']:.1f}%")
    print("="*80)
    
    return predictions.predictions, predictions.label_ids, power_results

def evaluate_with_test_dataset_with_power_monitoring(test_dataset, model=None, tokenizer=None, compute_metrics_func=None, save_results=True, output_dir="./evaluation_results", monitoring_interval=0.1):
    """Complete evaluation function with power monitoring"""
    
    # Load model and tokenizer if not provided
    if model is None or tokenizer is None:
        model, tokenizer = load_saved_model(model_path)
    
    # Set up label mappings
    global id2label, label2id
    # Use proper IOB2 format labels
    labels = label_list
    id2label = {i: label for i, label in enumerate(labels)}
    label2id = {label: i for i, label in enumerate(labels)}
        
    tokenized_test = test_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label2id),
        batched=True
    )
    
    # Evaluate on test set with power monitoring
    predictions, label_ids, power_results = evaluate_model_on_test_set_with_power_monitoring(
        model, tokenizer, tokenized_test, compute_metrics_func, monitoring_interval
    )
    
    # Compute F1 scores and get decoded sequences
    exact_f1, partial_f1, preds, labels = compute_advanced_f1_scores(predictions, label_ids)

    # Print unique gold and predicted labels (IOB tags) for quick inspection
    gold_labels = sorted({lab for seq in labels for lab in seq})
    pred_labels = sorted({lab for seq in preds for lab in seq})
    print("\nLabel sets (IOB):")
    print(f"Gold labels:      {gold_labels}")
    print(f"Predicted labels: {pred_labels}")

    # Print results
    print(f"\n{'='*60}")
    print("F1 SCORE RESULTS")
    print(f"{'='*60}")
    print(f"Exact F1 Score (Strict Entity Matching): {exact_f1:.4f}")
    print(f"Partial F1 Score (Lenient/Token-level): {partial_f1:.4f}")
    
    # Print detailed classification reports and get data for saving
    token_report, entity_report, stats, preds, labels = print_classification_reports(
        predictions, label_ids, "Test Set", save_reports=True
    )
    
    # Calculate carbon footprint
    carbon_footprint = estimate_carbon_footprint(power_results['total_energy_kwh'])
    
    # Save results to file if requested
    saved_files = None
    if save_results:
        # Prepare results dictionary with power monitoring data
        results_dict = {
            'evaluation_info': {
                'timestamp': datetime.now().isoformat(),
                'model_path': model_path,
                'test_dataset_size': len(test_dataset),
                'labels': list(id2label.values()) if id2label else [],
                'num_labels': len(id2label) if id2label else 0,
                'monitoring_interval': monitoring_interval
            },
            'f1_scores': {
                'exact_f1': float(exact_f1),
                'partial_f1': float(partial_f1)
            },
            'classification_reports': {
                'token_level': token_report,
                'entity_level': entity_report
            },
            'statistics': stats,
            'predictions': [pred for pred in preds],  # Convert to regular list
            'ground_truth': [label for label in labels],  # Convert to regular list
            'power_consumption': power_results,
            'carbon_footprint': carbon_footprint
        }
        
        # Save results to files
        saved_files = save_results_to_file(results_dict, output_dir)
    
    return exact_f1, partial_f1, predictions, label_ids, saved_files, power_results

# Quick evaluation function with power monitoring
def quick_evaluate_with_power_monitoring(save_results=True, output_dir="./evaluation_results", monitoring_interval=0.1):
    """Quick function to run evaluation with power monitoring"""
    print("Starting model evaluation with power monitoring...")
    
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = Dataset.from_json("/home/s27mhusa_hpc/Master-Thesis/Dataset19September/Test_NER_dataset_English.json")
    print(f"Test dataset loaded! Size: {len(test_dataset)}")
   
    # Load model and tokenizer
    print("Loading model...")
    model, tokenizer = load_saved_model(model_path)
    
    # Run evaluation with power monitoring
    exact_f1, partial_f1, predictions, label_ids, saved_files, power_results = evaluate_with_test_dataset_with_power_monitoring(
        test_dataset, model, tokenizer, save_results=save_results, output_dir=output_dir, monitoring_interval=monitoring_interval
    )
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"Exact F1 Score: {exact_f1:.4f}")
    print(f"Partial F1 Score: {partial_f1:.4f}")
    
    # Print power consumption summary
    print(f"\n⚡ POWER CONSUMPTION:")
    print(f"Total Energy: {power_results['total_energy_wh']:.4f} Wh")
    print(f"Average Power: {power_results['average_power_watts']:.2f} W")
    print(f"Duration: {power_results['duration_seconds']:.2f} seconds")
    
    # Print carbon footprint
    carbon_footprint = estimate_carbon_footprint(power_results['total_energy_kwh'])
    print(f"\n🌱 ENVIRONMENTAL IMPACT:")
    print(f"CO2 Emissions: {carbon_footprint['co2_g']:.2f} g")
    print(f"Trees needed to offset: {carbon_footprint['trees_needed_for_offset']:.4f}")
    
    if save_results and saved_files:
        print(f"\n💾 Results have been saved to: {output_dir}")
    
    return exact_f1, partial_f1, saved_files, power_results

if __name__ == "__main__":
    # Quick and easy way to run evaluation with power monitoring
    exact_f1, partial_f1, saved_files, power_results = quick_evaluate_with_power_monitoring(
        save_results=True, 
        output_dir="/home/s27mhusa_hpc/Master-Thesis/FineTuneCalculateResults/ScibertEnglishBroadWithPower",
        monitoring_interval=0.1
    )
    
    print(f"\n✅ Evaluation with power monitoring completed!")
    print(f"📊 Exact F1: {exact_f1:.4f}")
    print(f"📈 Partial F1: {partial_f1:.4f}")
    print(f"⚡ Total Energy: {power_results['total_energy_wh']:.4f} Wh")
    print(f"🌱 CO2 Emissions: {estimate_carbon_footprint(power_results['total_energy_kwh'])['co2_g']:.2f} g")
    
    if saved_files:
        print(f"\n📁 All results saved to files!")
        print("Check the evaluation_results folder for detailed outputs including power consumption data.")
