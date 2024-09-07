#!/bin/bash
#
#SBATCH --job-name=extract_query_awe_embeddings
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=tc062-pool3
#SBATCH --time=01:00:00
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
PHONE_TIMINGS_FNAME=$2
MIN_PHONE_SEQ_LENGTH=$3
MAX_PHONE_SEQ_LENGTH=$4
USE_WINDOW_SPLITTER_INSTEAD_OF_PHONE_SPLITTER=$5
LAYER=$6
SAVE_EMBEDDING_FOLDER=$7
MODEL_SAVE_DIR=$8
MODEL_NAME=$9

srun python -u awe_model/extract_query_doc_awe_embeddings.py queries 1 0 $LANGUAGE $PHONE_TIMINGS_FNAME $MIN_PHONE_SEQ_LENGTH $MAX_PHONE_SEQ_LENGTH $USE_WINDOW_SPLITTER_INSTEAD_OF_PHONE_SPLITTER $LAYER $SAVE_EMBEDDING_FOLDER $MODEL_SAVE_DIR $MODEL_NAME