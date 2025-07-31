#!/bin/bash
#SBATCH --partition=sgpu_short
#SBATCH --time=8:00:00
#SBATCH --gpus=2
#SBATCH --output=gpu_job_%j.out

MODEL_NAME=$1

INPUT_DIR="/home/s27mhusa_hpc/Master-Thesis/Text_Files_For_LLM_Input"
OUTPUT_DIR="/home/s27mhusa_hpc/Master-Thesis/Results/Results_modified_prompt/LLM_annotated_${MODEL_NAME}"
OUTPUT_DIR_JSON="/home/s27mhusa_hpc/Master-Thesis/Results/Results_modified_prompt_json/LLM_annotated_${MODEL_NAME}"
MODEL_PATH="/lustre/scratch/data/s27mhusa_hpc-murtuza_master_thesis/${MODEL_NAME}"

module load CUDA/12.6.0

source ~/.bashrc
conda init
conda deactivate 
conda activate Llama

# Record start time
start_time=$(date +%s)

# Start GPU usage logging in the background
GPU_LOG_FILE="gpu_usage_${SLURM_JOB_ID}.log"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 10 > "$GPU_LOG_FILE" &
nvidia_pid=$!

# Run your Python script
python /home/s27mhusa_hpc/Master-Thesis/Zero-Shot-Prompting/NER_Prompting_generic.py \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --output_dir_json "$OUTPUT_DIR_JSON" \
  --model_name "$MODEL_NAME" \
  --model_path "$MODEL_PATH" \
  --max_length 1512

# Kill the GPU logging process
kill $nvidia_pid

# Record end time
end_time=$(date +%s)

# Compute and print elapsed time
elapsed=$(( end_time - start_time ))
echo "Total GPU execution time: $elapsed seconds"
