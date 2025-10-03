#!/bin/bash
sbatch job-embeddings_16.sbatch Qwen2.5-72B-Instruct 5
sbatch job-embeddings_16.sbatch Qwen2.5-72B-Instruct 4
sbatch job-embeddings_16.sbatch Qwen2.5-72B-Instruct 3
sbatch job-embeddings_16.sbatch Qwen2.5-72B-Instruct 2
sbatch job-embeddings_16.sbatch Qwen2.5-72B-Instruct 1
sbatch job-embeddings_16.sbatch Llama-3.3-70B-Instruct 5
sbatch job-embeddings_16.sbatch Llama-3.3-70B-Instruct 4
sbatch job-embeddings_16.sbatch Llama-3.3-70B-Instruct 3
sbatch job-embeddings_16.sbatch Llama-3.3-70B-Instruct 2
sbatch job-embeddings_16.sbatch Llama-3.3-70B-Instruct 1

