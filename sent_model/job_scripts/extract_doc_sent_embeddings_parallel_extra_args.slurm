#!/bin/bash
#
#SBATCH --job-name=extract_doc_sent_embeddings
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=tc062-pool3
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-4

echo $SLURM_ARRAY_TASK_ID

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
USE_AWES=$6

srun python -u sent_model/extract_query_doc_sent_embeddings.py documents 5 $SLURM_ARRAY_TASK_ID $LANGUAGE $LAYER $SAVE_EMBEDDING_FOLDER $MODEL_SAVE_DIR $MODEL_NAME $USE_AWES