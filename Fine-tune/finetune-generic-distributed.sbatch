#!/bin/bash
#SBATCH --partition=mlgpu_devel
#SBATCH --time=1:00:00
#SBATCH --gpus=16        
#SBATCH --output=/home/s27mhusa_hpc/Master-Thesis/OutputNewDatasets/finetune_gpu_mdeberta-v3-base_job_%j.out

MODEL_NAME=$1

module load CUDA/12.6.0

source ~/.bashrc
conda init
conda deactivate 
conda activate Llama

# Clear any existing distributed environment variables
unset RANK
unset WORLD_SIZE
unset LOCAL_RANK
unset MASTER_ADDR
unset MASTER_PORT

# Print environment info
echo "=== Environment Information ==="
echo "Node: $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "==============================="

# Print GPU information
echo "=== GPU Information ==="
nvidia-smi --list-gpus
nvidia-smi
echo "======================="

# Record start time
start_time=$(date +%s)

# === Begin Resource Monitoring ===
LOG_DIR="/home/s27mhusa_hpc/Master-Thesis/OutputNewDatasets/job_monitor_logs_${MODEL_NAME}_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"
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

# === Determine training method based on GPU count ===
SCRIPT_PATH="/home/s27mhusa_hpc/Master-Thesis/Fine-tune/fine_tune_mdeberta_distributed.py"

if [ -z "$SLURM_GPUS_PER_NODE" ]; then
    # Fallback: detect GPUs manually
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
else
    NUM_GPUS=$SLURM_GPUS_PER_NODE
fi

echo "Number of GPUs detected: $NUM_GPUS"

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "=== Starting Multi-GPU Distributed Training ==="
    echo "Using torchrun with $NUM_GPUS GPUs"
    
    # Set master address and port for distributed training
    export MASTER_ADDR=$(hostname)
    export MASTER_PORT=29500
    
    echo "MASTER_ADDR: $MASTER_ADDR"
    echo "MASTER_PORT: $MASTER_PORT"
    
    # Use torchrun for distributed training
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        --nnodes=1 \
        --node_rank=0 \
        $SCRIPT_PATH
    
    TRAIN_EXIT_CODE=$?
else
    echo "=== Starting Single GPU Training ==="
    echo "Using standard python execution"
    
    python $SCRIPT_PATH
    TRAIN_EXIT_CODE=$?
fi

echo "Training completed with exit code: $TRAIN_EXIT_CODE"

# === Stop Resource Monitoring ===
echo "Stopping resource monitoring..."
kill $GPU_PID 2>/dev/null || true
kill $CPU_PID 2>/dev/null || true
kill $IO_PID 2>/dev/null || true

# Wait a moment for processes to terminate
sleep 2

# Record end time
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))

echo "Total execution time: $elapsed seconds"
echo "$elapsed" > "$LOG_DIR/elapsed_time_seconds.txt"
echo "$TRAIN_EXIT_CODE" > "$LOG_DIR/exit_code.txt"

# === SLURM Summary Metrics ===
echo "Fetching SLURM job summary..."
sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,MaxRSS,CPUTime,ExitCode > "$LOG_DIR/slurm_summary.log" 2>/dev/null || echo "sacct not available"
if [ -f "$LOG_DIR/slurm_summary.log" ]; then
    cat "$LOG_DIR/slurm_summary.log"
fi

echo "=== Job Summary ==="
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs used: $NUM_GPUS"
echo "Exit code: $TRAIN_EXIT_CODE"
echo "Duration: $elapsed seconds"
echo "Logs saved in: $LOG_DIR"
echo "=================="

exit $TRAIN_EXIT_CODE