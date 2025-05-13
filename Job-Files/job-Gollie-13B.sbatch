#!/bin/bash
#SBATCH --partition=sgpu_short
#SBATCH --time=8:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1



module load CUDA/12.6.0


source ~/.bashrc
conda init
conda deactivate 
conda activate Llama

python /home/s27mhusa_hpc/pilot-uc-textmining-metadata/code/GoLLIE/NER_Prompting-GoLLIE-13B.py