#!/bin/bash
#
#SBATCH --job-name=hubert_extraction
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=tc062-pool3
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

export HF_HOME=/work/tc062/tc062/sanjayb/.cache/hugging_face_cache
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export PYTHONPATH=/work/tc062/tc062/sanjayb/voice-search-project:$PYTHONPATH

# Load the required modules 
module load python/3.10.8-gpu
module load pytorch/1.13.1-gpu
source /work/tc062/tc062/sanjayb/voice-search-project/mhubert_model/hubert_env_gpu/bin/activate

srun python -u mhubert_model/extract_mHuBERT_embeddings.py