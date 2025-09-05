#!/bin/bash
#SBATCH --partition=sgpu_medium
#SBATCH --time=24:00:00
#SBATCH --gpus=4
#SBATCH --output=/home/s27mhusa_hpc/Master-Thesis/Output-FineTune-Document-12Aug-Prompting/gpu_job_FineTune_Prompting_Document_%j_12-8-2025.out

MODEL_NAME=$1

JSONL_PATH="/home/s27mhusa_hpc/Master-Thesis/Dataset-25-July-Document-Level/Test_ner_dataset_document_input_output.jsonl"
OUTPUT_PRED="/home/s27mhusa_hpc/Master-Thesis/LLM-Predictions-Document-12Aug/Test_predictions_${MODEL_NAME}-Document.jsonl"
OUTPUT_GOLD="/home/s27mhusa_hpc/Master-Thesis/LLM-Predictions-Document-12Aug/Test_gold_${MODEL_NAME}-Document.jsonl"
MODEL_PATH="/lustre/scratch/data/s27mhusa_hpc-murtuza_master_thesis/${MODEL_NAME}"

module load CUDA/12.6.0
module load sysstat


source ~/.bashrc
conda init
conda deactivate 
conda activate autotrain

# Record start time
start_time=$(date +%s)

# === Begin Resource Monitoring ===
LOG_DIR="/home/s27mhusa_hpc/Master-Thesis/Output-FineTune-Document-12Aug-Prompting/job_monitor_logs_${MODEL_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
INTERVAL=30

echo "Starting resource monitoring every $INTERVAL seconds..."
echo "Logs will be stored in: $LOG_DIR"

# GPU Monitoring
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,temperature.gpu,power.draw \
  --format=csv -l $INTERVAL > "$LOG_DIR/gpu_usage.log" &
GPU_PID=$!

# CPU & Memory Monitoring
(
  while true; do
    echo "$(date)" >> "$LOG_DIR/cpu_mem.log"
    top -b -n1 | head -n 20 >> "$LOG_DIR/cpu_mem.log"
    echo "===============================" >> "$LOG_DIR/cpu_mem.log"
    sleep $INTERVAL
  done
) &
CPU_PID=$!

# Disk I/O Monitoring
iostat -xz $INTERVAL > "$LOG_DIR/disk_io.log" &
IO_PID=$!

# Save monitor PIDs
echo $GPU_PID > "$LOG_DIR/pids.txt"
echo $CPU_PID >> "$LOG_DIR/pids.txt"
echo $IO_PID >> "$LOG_DIR/pids.txt"

# === Run Main Script ===
python /home/s27mhusa_hpc/Master-Thesis/Zero-Shot-Prompting/NER_Prompting_generic_chat-finetune-llm.py \
  --jsonl_path "$JSONL_PATH" \
  --output_pred "$OUTPUT_PRED" \
  --output_gold "$OUTPUT_GOLD" \
  --model_name "$MODEL_NAME" \
  --model_path "$MODEL_PATH" \
  --max_length 2048

# === Stop Resource Monitoring ===
echo "Stopping resource monitoring..."
kill $GPU_PID
kill $CPU_PID
kill $IO_PID

# Record end time
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))

echo "Total GPU execution time: $elapsed seconds"
echo "$elapsed" > "$LOG_DIR/elapsed_time_seconds.txt"

# === SLURM Summary Metrics ===
echo "Fetching SLURM job summary..."
sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,MaxRSS,CPUTime,ExitCode > "$LOG_DIR/slurm_summary.log"
cat "$LOG_DIR/slurm_summary.log"

echo "All logs are saved in: $LOG_DIR"
