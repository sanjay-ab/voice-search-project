#!/bin/bash
#
#SBATCH --job-name=hubert_pool_embeddings
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --account=tc062-pool3
#SBATCH --time=01:00:00
#SBATCH --mem=64g

export HF_HOME=/work/tc062/tc062/sanjayb/.cache/hugging_face_cache
export PYTHONPATH=/work/tc062/tc062/sanjayb/voice-search-project:$PYTHONPATH

# Load the required modules 
module load python/3.9.13
module load pytorch/1.13.1
source /work/tc062/tc062/sanjayb/voice-search-project/mhubert_model/hubert_env/bin/activate

WINDOW_SIZE_MS=$1
STRIDE_MS=$2
LAYER=$3
RUN_FOR_QUERIES=$4
RUN_FOR_DOCUMENTS=$5
LANGUAGE=$6
USE_QUERIES_CUT_AFTER_EMBEDDING=$7

srun python -u mhubert_model/pool_mHuBERT_embeddings.py $WINDOW_SIZE_MS $STRIDE_MS $LAYER $RUN_FOR_QUERIES $RUN_FOR_DOCUMENTS 1 0 $LANGUAGE $USE_QUERIES_CUT_AFTER_EMBEDDING