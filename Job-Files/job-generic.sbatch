#!/bin/bash
#SBATCH --partition=sgpu_short
#SBATCH --time=8:00:00
#SBATCH --gpus=4


MODEL_NAME=$1

INPUT_DIR="/home/s27mhusa_hpc/Master-Thesis/Text_Files_For_LLM_Input"
OUTPUT_DIR="/home/s27mhusa_hpc/Master-Thesis/Results/Results_new_prompt/LLM_annotated_${MODEL_NAME}"
MODEL_PATH="/lustre/scratch/data/s27mhusa_hpc-murtuza_master_thesis/${MODEL_NAME}"

module load CUDA/12.6.0


source ~/.bashrc
conda init
conda deactivate 
conda activate Llama

python /home/s27mhusa_hpc/Master-Thesis/Zero-Shot-Prompting/NER_Prompting_generic.py \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --model_name "$MODEL_NAME" \
  --model_path "$MODEL_PATH" \
  --max_length 1512