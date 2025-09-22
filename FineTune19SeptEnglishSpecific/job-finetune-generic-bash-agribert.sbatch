#!/bin/bash
#SBATCH --partition=mlgpu_short
#SBATCH --time=8:00:00
#SBATCH --gpus=16
#SBATCH --output=/home/s27mhusa_hpc/Master-Thesis/OutputNewDatasets22thSeptemberFineTune/finetune_gpu_agribert_specific_english_3.0_%j.out

MODEL_NAME=$1


module load CUDA/12.6.0

source ~/.bashrc
conda init
conda deactivate 
conda activate Llama

# Record start time
start_time=$(date +%s)

# === Begin Resource Monitoring ===
LOG_DIR="/home/s27mhusa_hpc/Master-Thesis/OutputNewDatasets22thSeptemberFineTuneEnglish/job_monitor_logs_${MODEL_NAME}_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"
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
python /home/s27mhusa_hpc/Master-Thesis/FineTune19SeptEnglishSpecific/fine-tune-agribert-reg_broad.py

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
