#!/bin/bash
#
#SBATCH --job-name=extract_doc_query_awe_embeddings
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --account=tc062-pool3
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --mem=64g

export HF_HOME=/work/tc062/tc062/sanjayb/.cache/hugging_face_cache
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export PYTHONPATH=/work/tc062/tc062/sanjayb/dissertation:$PYTHONPATH

# Load the required modules 
module load python/3.9.13
module load pytorch/1.13.1
source /work/tc062/tc062/sanjayb/dissertation/mhubert_model/hubert_env/bin/activate

srun python -u awe_model/extract_query_doc_awe_embeddings.py queries 1 0