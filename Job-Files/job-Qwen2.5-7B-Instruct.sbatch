#!/bin/bash
#SBATCH --partition=mlgpu_short
#SBATCH --time=8:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1



module load CUDA/12.6.0


source ~/.bashrc
conda init
conda deactivate 
conda activate Llama

python /home/s27mhusa_hpc/Master-Thesis/Zero-Shot-Prompting/NER_Prompting-Qwen2.5-7B-Instruct.py