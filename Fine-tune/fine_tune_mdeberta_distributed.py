import numpy as np
import evaluate
import wandb
import gc
from datasets import Dataset, DatasetDict
from datasets import config

from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import classification_report
from seqeval.metrics import classification_report as seqeval_classification_report
import os
import torch
import torch.distributed as dist
import sys
import traceback
import time
import socket

# Set a new cache directory
cache_dir = "/lustre/scratch/data/s27mhusa_hpc-murtuza_master_thesis/new_hf_cache_1"
os.makedirs(cache_dir, exist_ok=True)

# Configure datasets to use the new cache
config.CACHE_ROOT = cache_dir
os.environ['HF_DATASETS_CACHE'] = cache_dir

# Set environment variables for better NCCL stability
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NCCL_DEBUG"] = "INFO"  # Add NCCL debugging
os.environ["NCCL_TIMEOUT"] = "1800"  # 30 minutes timeout
os.environ["NCCL_IB_DISABLE"] = "1"  # Disable InfiniBand if causing issues
os.environ["NCCL_P2P_DISABLE"] = "1"  # Disable P2P if causing issues
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # For better error messages

# Global variables
label_list = ["O","B-soilReferenceGroup","I-soilReferenceGroup", "B-soilOrganicCarbon", "I-soilOrganicCarbon", "B-soilTexture", "I-soilTexture", "B-startTime", "I-startTime", "B-endTime", "I-endTime", "B-city", "I-city", "B-duration", "I-duration", "B-cropSpecies", "I-cropSpecies", "B-soilAvailableNitrogen", "I-soilAvailableNitrogen", "B-soilDepth", "I-soilDepth", "B-region", "I-region", "B-country", "I-country", "B-longitude", "I-longitude", "B-latitude", "I-latitude", "B-cropVariety", "I-cropVariety", "B-soilPH", "I-soilPH", "B-soilBulkDensity", "I-soilBulkDensity"]
label_to_id = {l: i for i, l in enumerate(label_list)}

def get_free_port():
    """Find a free port for distributed training"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def setup_distributed():
    """Initialize distributed training with robust error handling"""
    try:
        # Get environment variables
        rank_env = os.environ.get("RANK", "")
        world_size_env = os.environ.get("WORLD_SIZE", "")
        local_rank_env = os.environ.get("LOCAL_RANK", "")
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = os.environ.get("MASTER_PORT", "")
        
        print(f"Environment variables:")
        print(f"  RANK='{rank_env}'")
        print(f"  WORLD_SIZE='{world_size_env}'")
        print(f"  LOCAL_RANK='{local_rank_env}'")
        print(f"  MASTER_ADDR='{master_addr}'")
        print(f"  MASTER_PORT='{master_port}'")
        
        # Check if we should use distributed training
        if not (rank_env and world_size_env and rank_env.strip() and world_size_env.strip()):
            print("Distributed environment variables not found. Using single GPU training.")
            return 0, 0, False
            
        rank = int(rank_env)
        world_size = int(world_size_env)
        local_rank = int(local_rank_env) if local_rank_env.strip() else rank % torch.cuda.device_count()
        
        print(f"Parsed values: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        
        # Validate CUDA availability
        if not torch.cuda.is_available():
            print("CUDA is not available. Cannot use distributed training.")
            return 0, 0, False
            
        if local_rank >= torch.cuda.device_count():
            print(f"Error: local_rank {local_rank} >= available GPUs {torch.cuda.device_count()}")
            return 0, 0, False
        
        # Set the device early
        torch.cuda.set_device(local_rank)
        print(f"Set CUDA device to {local_rank}")
        
        # Set master port if not set
        if not master_port:
            if rank == 0:
                port = get_free_port()
                os.environ["MASTER_PORT"] = str(port)
                print(f"Set MASTER_PORT to {port}")
            else:
                # Wait a bit for rank 0 to set the port
                time.sleep(2)
                port = os.environ.get("MASTER_PORT", "29500")
        else:
            port = master_port
            
        print(f"Using MASTER_ADDR={master_addr}, MASTER_PORT={port}")
        
        # Check if process group is already initialized
        if dist.is_initialized():
            print("Process group already initialized")
            return rank, local_rank, True
            
        # Initialize the process group with timeout
        print(f"Initializing process group for rank {rank}/{world_size}")
        
        # Try different backends if NCCL fails
        backends = ["nccl"]  # We can add "gloo" as fallback if needed
        
        for backend in backends:
            try:
                print(f"Trying backend: {backend}")
                dist.init_process_group(
                    backend=backend,
                    init_method=f"tcp://{master_addr}:{port}",
                    rank=rank,
                    world_size=world_size,
                    timeout=torch.distributed.timedelta(seconds=1800)  # 30 minutes
                )
                print(f"Successfully initialized process group with {backend}")
                break
            except Exception as e:
                print(f"Failed to initialize with {backend}: {e}")
                if backend == backends[-1]:  # Last backend
                    raise e
                continue
        
        # Verify the setup with a simple all-reduce operation
        print("Testing distributed communication...")
        test_tensor = torch.tensor([rank], dtype=torch.float32).cuda()
        dist.all_reduce(test_tensor)
        expected_sum = sum(range(world_size))
        
        if abs(test_tensor.item() - expected_sum) < 1e-6:
            print(f"Distributed communication test passed. Sum: {test_tensor.item()}")
        else:
            print(f"Distributed communication test failed. Expected: {expected_sum}, Got: {test_tensor.item()}")
            raise RuntimeError("Distributed communication test failed")
            
        print(f"Successfully initialized distributed training on GPU {local_rank}")
        return rank, local_rank, True
        
    except Exception as e:
        print(f"Error in setup_distributed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        print("Falling back to single GPU training")
        
        # Clean up any partial initialization
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except:
            pass
            
        return 0, 0, False

def cleanup_distributed(is_distributed):
    """Clean up distributed training"""
    if is_distributed and dist.is_initialized():
        try:
            print("Cleaning up distributed training...")
            dist.barrier()  # Synchronize all processes
            dist.destroy_process_group()
            print("Distributed training cleanup completed.")
        except Exception as e:
            print(f"Error during cleanup: {e}")

def print_gpu_info(rank):
    """Print GPU information"""
    if rank != 0:  # Only print from rank 0
        return
        
    try:
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            print(f"Current device: {current_device}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
                print(f"  Compute Capability: {props.major}.{props.minor}")
    except Exception as e:
        print(f"Error getting GPU info: {e}")

def tokenize_and_align_labels(examples):
    """Tokenize and align labels with error handling"""
    try:
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding="max_length",
            max_length=1512,  # Further reduced to prevent memory issues
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
                    if word_idx < len(label):
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(-100)  # Handle edge case
                else:
                    if word_idx < len(label):
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(-100)  # Handle edge case
                previous_word_idx = word_idx
            
            labels.append(label_ids)
        
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    except Exception as e:
        print(f"Error in tokenize_and_align_labels: {e}")
        raise

def align_predictions(predictions, label_ids):
    """Align predictions with labels"""
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
    """Compute evaluation metrics"""
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

def main():
    global tokenizer, seqeval
    
    rank, local_rank, is_distributed = None, None, False
    
    try:
        print("Starting distributed training setup...")
        
        # Setup distributed training
        rank, local_rank, is_distributed = setup_distributed()
        
        print(f"Process {rank}: Training setup - distributed: {is_distributed}")
        
        # Print GPU info only on main process
        print_gpu_info(rank)
        
        # Define seqeval
        seqeval = evaluate.load("seqeval")
        
        # Only login to wandb on the main process
        if rank == 0:
            try:
                wandb.login(key="ed7faaa7784428261467aee38c86ccc5c316f954")
                print("Successfully logged into wandb")
            except Exception as e:
                print(f"Warning: Failed to login to wandb: {e}")
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Load dataset
        print(f"Process {rank}: Loading datasets...")
        
        try:
            train_dataset = Dataset.from_json("/home/s27mhusa_hpc/Master-Thesis/NewDatasets27August/Train_ner_dataset_sentence_filtered_train_stratified.json")
            val_dataset   = Dataset.from_json("/home/s27mhusa_hpc/Master-Thesis/NewDatasets27August/Train_ner_dataset_sentence_filtered_val_stratified.json")
            test_dataset  = Dataset.from_json("/home/s27mhusa_hpc/Master-Thesis/NewDatasets27August/Test_ner_dataset_sentence.json")

            dataset = DatasetDict({
                "train": train_dataset,
                "validation": val_dataset,
                "test": test_dataset
            })
            
            if rank == 0:
                print(f"Loaded datasets - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
                
        except Exception as e:
            print(f"Error loading datasets: {e}")
            raise

        # Initialize tokenizer
        model_checkpoint = "microsoft/mdeberta-v3-base"
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, cache_dir=cache_dir)
        
        if rank == 0:
            print(f"Loaded tokenizer: {model_checkpoint}")

        # Tokenize dataset
        print(f"Process {rank}: Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            tokenize_and_align_labels,
            batched=True,
            batch_size=32,  # Reduced batch size
            remove_columns=dataset["train"].column_names,
            num_proc=1,  # Single process to avoid issues
            load_from_cache_file=True,  # Use cache if available
        )
        
        if rank == 0:
            print("Dataset tokenization completed")

        # Use fixed hyperparameters
        if is_distributed:
            per_device_batch_size = 2  # Very small for distributed
            gradient_accumulation = 32  # Large accumulation
            num_workers = 0  # No additional workers
        else:
            per_device_batch_size = 2  # Small for single GPU
            gradient_accumulation = 16
            num_workers = 0
        
        best_params = {
            "learning_rate": 4e-5,
            "num_train_epochs": 15,  # Reduced for testing
            "per_device_train_batch_size": per_device_batch_size,
            "weight_decay": 0.03
        }
        
        if rank == 0:
            print(f"Using hyperparameters: {best_params}")
            print(f"Effective batch size: {per_device_batch_size * gradient_accumulation * (torch.distributed.get_world_size() if is_distributed else 1)}")

        # Training arguments
        training_args = TrainingArguments(
            output_dir="/lustre/scratch/data/s27mhusa_hpc-murtuza_master_thesis/mdeberta_ner_model_robust",
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_dir="./logs_robust",
            run_name="mdeberta-v3-base_robust_run" if rank == 0 else None,
            
            # Core training parameters
            learning_rate=best_params["learning_rate"],
            num_train_epochs=best_params["num_train_epochs"],
            per_device_train_batch_size=best_params["per_device_train_batch_size"],
            per_device_eval_batch_size=best_params["per_device_train_batch_size"],
            weight_decay=best_params["weight_decay"],
            
            # Optimization settings
            gradient_accumulation_steps=gradient_accumulation,
            max_grad_norm=1.0,
            warmup_ratio=0.1,
            lr_scheduler_type="linear",
            fp16=True,
            
            # Evaluation and saving
            metric_for_best_model="f1",
            greater_is_better=True,
            load_best_model_at_end=True,
            save_total_limit=1,  # Save space
            save_on_each_node=False,
            
            # Data loading
            dataloader_pin_memory=False,
            dataloader_num_workers=num_workers,
            dataloader_drop_last=True,  # Always drop last for stability
            
            # Distributed settings
            ddp_backend="nccl" if is_distributed else None,
            ddp_find_unused_parameters=False if is_distributed else None,
            ddp_bucket_cap_mb=25 if is_distributed else None,  # Reduce bucket size
            
            # Logging and reporting
            logging_steps=50,
            logging_first_step=True,
            report_to="wandb" if rank == 0 else None,
            
            # Stability settings
            ignore_data_skip=True,
            remove_unused_columns=True,
        )

        def model_init():
            """Initialize model with error handling"""
            try:
                model = AutoModelForTokenClassification.from_pretrained(
                    model_checkpoint,
                    num_labels=len(label_list),
                    cache_dir=cache_dir
                )
                return model
            except Exception as e:
                print(f"Error initializing model: {e}")
                raise

        # Create trainer
        trainer = Trainer(
            model_init=model_init,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]  # Reduced patience
        )

        # Synchronize before training
        if is_distributed:
            dist.barrier()
            print(f"Process {rank}: All processes synchronized, starting training...")

        # Train the model
        if rank == 0:
            print("Starting training...")
        
        trainer.train()
        
        # Synchronize after training
        if is_distributed:
            dist.barrier()
            
        # Only evaluate on main process
        if rank == 0:
            print("\nEvaluating on validation data...")
            val_outputs = trainer.predict(tokenized_dataset["validation"])
            
            print("\nEvaluating on test data...")
            test_results = trainer.predict(tokenized_dataset["test"])
            print("\nTest Metrics:")
            print(test_results.metrics)
            
            print("\n" + "="*60)
            print("TRAINING COMPLETED SUCCESSFULLY!")
            print("="*60)

        # Final synchronization
        if is_distributed:
            dist.barrier()
                
    except Exception as e:
        print(f"Fatal error in main: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return_code = 1
    else:
        return_code = 0
    finally:
        # Always clean up
        cleanup_distributed(is_distributed)
        
    sys.exit(return_code)

if __name__ == "__main__":
    main()