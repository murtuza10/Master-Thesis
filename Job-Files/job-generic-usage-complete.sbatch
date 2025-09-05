#!/bin/bash
#SBATCH --partition=mlgpu_devel
#SBATCH --time=1:00:00
#SBATCH --gpus=1
#SBATCH --output=/home/s27mhusa_hpc/Master-Thesis/Output-23thAugust-DeepSeek/gpu_job_TestFiles_%j_23thAugust.out

MODEL_NAME=$1

INPUT_DIR="/home/s27mhusa_hpc/Master-Thesis/Text_Files_For_LLM_Input"
OUTPUT_DIR="/home/s27mhusa_hpc/Master-Thesis/Results/Results_TestFiles_23thAugust_DeepSeek/LLM_annotated_${MODEL_NAME}"
OUTPUT_DIR_JSON="/home/s27mhusa_hpc/Master-Thesis/Results/Results_TestFiles_23thAugust_DeepSeek_json/LLM_annotated_${MODEL_NAME}"
MODEL_PATH="/lustre/scratch/data/s27mhusa_hpc-murtuza_master_thesis/${MODEL_NAME}"

module load CUDA/12.6.0
module load sysstat


source ~/.bashrc
conda init
conda deactivate 
conda activate Llama

# Record start time
start_time=$(date +%s)

# === Begin Resource Monitoring ===
LOG_DIR="/home/s27mhusa_hpc/Master-Thesis/Output-23thAugust-DeepSeek/job_monitor_logs_TestFiles${MODEL_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
INTERVAL=1

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
python /home/s27mhusa_hpc/Master-Thesis/Zero-Shot-Prompting/NER_Prompting_chat_deepseekV3-5.py \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --output_dir_json "$OUTPUT_DIR_JSON" \
  --max_length 1512

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
