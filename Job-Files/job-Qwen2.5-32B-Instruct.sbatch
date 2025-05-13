#!/bin/bash
#SBATCH --partition=mlgpu_short
#SBATCH --time=8:00:00
#SBATCH --gpus=2
#SBATCH --ntasks=1



module load CUDA/12.6.0


source ~/.bashrc
conda init
conda deactivate 
conda activate Llama

python NER_Prompting-Qwen2.5-32B-Instruct.py