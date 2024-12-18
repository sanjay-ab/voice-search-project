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
USE_QUERIES_CUT_AFTER_EMBEDDING=$6
LAYER=$7
SAVE_EMBEDDING_FOLDER=$8
MODEL_SAVE_DIR=$9
MODEL_NAME=${10}

QUERIES_DIR_NAME="queries"
if [[ "$LANGUAGE" != "banjara" ]]; then
    if [[ "$USE_QUERIES_CUT_AFTER_EMBEDDING" == "True" ]]; then
        QUERIES_DIR_NAME="queries_cut_after_embedding"
    elif [[ "$USE_QUERIES_CUT_AFTER_EMBEDDING" == "False" ]]; then
        QUERIES_DIR_NAME="queries"
    else
        echo "USE_QUERIES_CUT_AFTER_EMBEDDING must be either True or False"
        exit 1
    fi
fi

srun python -u awe_model/extract_query_doc_awe_embeddings.py $QUERIES_DIR_NAME 1 0 $LANGUAGE $PHONE_TIMINGS_FNAME $MIN_PHONE_SEQ_LENGTH $MAX_PHONE_SEQ_LENGTH $USE_WINDOW_SPLITTER_INSTEAD_OF_PHONE_SPLITTER $LAYER $SAVE_EMBEDDING_FOLDER $MODEL_SAVE_DIR $MODEL_NAME