#!/bin/bash
#SBATCH --partition=sgpu_short
#SBATCH --time=8:00:00
#SBATCH --gpus=2
#SBATCH --ntasks=2



module load CUDA/12.6.0


source ~/.bashrc
conda init
conda deactivate 
conda activate Llama

python /home/s27mhusa_hpc/Master-Thesis/Zero-Shot-Prompting/NER_Prompting-Qwen2.5-72B-Instruct.py