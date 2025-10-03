#!/bin/bash
sbatch job-embeddings_openAI.sbatch GPT5 5
sbatch job-embeddings_gpt5.sbatch GPT5 4
sbatch job-embeddings_gpt5.sbatch GPT5 3
sbatch job-embeddings_gpt5.sbatch GPT5 2
sbatch job-embeddings_gpt5.sbatch GPT5 1

    
