#!/bin/bash
#
#SBATCH --job-name=ranking_vectorised
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=tc062-pool3
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

export HF_HOME=/work/tc062/tc062/sanjayb/.cache/hugging_face_cache
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export PYTHONPATH=/work/tc062/tc062/sanjayb/voice-search-project:$PYTHONPATH

# Load the required modules 
module load python/3.10.8-gpu
module load pytorch/1.13.1-gpu
source /work/tc062/tc062/sanjayb/voice-search-project/mhubert_model/hubert_env_gpu/bin/activate

LANGUAGE=$1
LAYER=$2
MIN_PHONE_SEQ_LENGTH=$3
MAX_PHONE_SEQ_LENGTH=$4
USE_QUERIES_CUT_AFTER_EMBEDDING=$5
DOC_QUERY_SUFFIX=$6
RESULTS_PATH=$7

srun python -u awe_model/query_document_search_vectorised.py $LANGUAGE $LAYER $MIN_PHONE_SEQ_LENGTH $MAX_PHONE_SEQ_LENGTH $USE_QUERIES_CUT_AFTER_EMBEDDING $DOC_QUERY_SUFFIX $RESULTS_PATH