#!/bin/bash
#SBATCH --partition=mlgpu_long
#SBATCH --time=144:00:00
#SBATCH --gpus=4
#SBATCH --output=gpu_job_weights%j.out


module load CUDA/12.6.0
module load sysstat


source ~/.bashrc
conda init
conda deactivate 
conda activate Llama

# === Run Main Script ===
python /lustre/scratch/data/s27mhusa_hpc-murtuza_master_thesis/DeepSeek-V3/inference/fp8_cast_bf16.py --input-fp8-hf-path /lustre/scratch/data/s27mhusa_hpc-murtuza_master_thesis/DeepSeek-V3 --output-bf16-hf-path /lustre/scratch/data/s27mhusa_hpc-murtuza_master_thesis/DeepSeek-V3-bf16