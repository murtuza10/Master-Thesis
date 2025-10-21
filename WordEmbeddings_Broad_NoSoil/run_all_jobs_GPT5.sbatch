#!/bin/bash
sbatch job-embeddings_openAI.sbatch gpt-5 5
sbatch job-embeddings_gpt5.sbatch gpt-5 4
sbatch job-embeddings_gpt5.sbatch gpt-5 3
sbatch job-embeddings_gpt5.sbatch gpt-5 2
sbatch job-embeddings_gpt5.sbatch gpt-5 1

    
