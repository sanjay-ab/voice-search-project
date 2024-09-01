#!/bin/bash
#
#SBATCH --job-name=extract_query_rec_embeddings
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=tc062-pool3
#SBATCH --time=01:00:00
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
SAVE_EMBEDDING_FOLDER=$3
MODEL_SAVE_DIR=$4
MODEL_NAME=$5
OUTPUT_DIM=$6
MODEL_TYPE=$7

srun python -u rec_model/extract_query_doc_rec_embeddings.py queries 1 0 $LANGUAGE $LAYER $SAVE_EMBEDDING_FOLDER $MODEL_SAVE_DIR $MODEL_NAME $OUTPUT_DIM $MODEL_TYPE