#!/bin/bash
#
#SBATCH --job-name=ranking_vectorised
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=tc062-pool3
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

export HF_HOME=/work/tc062/tc062/sanjayb/.cache/hugging_face_cache
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export PYTHONPATH=/work/tc062/tc062/sanjayb/dissertation:$PYTHONPATH

# Load the required modules 
module load python/3.10.8-gpu
module load pytorch/1.13.1-gpu
source /work/tc062/tc062/sanjayb/dissertation/mhubert_model/hubert_env_gpu/bin/activate

LANGUAGE=$1
LAYER=$2
DOC_QUERY_SUFFIX=$3
RESULTS_PATH=$4

srun python -u rec_model/query_document_search_vectorised.py $LANGUAGE $LAYER $DOC_QUERY_SUFFIX $RESULTS_PATH