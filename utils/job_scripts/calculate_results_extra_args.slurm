#!/bin/bash
#
#SBATCH --job-name=calculate_results
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --account=tc062-pool3
#SBATCH --time=00:20:00
#SBATCH --nodes=1

export HF_HOME=/work/tc062/tc062/sanjayb/.cache/hugging_face_cache
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export PYTHONPATH=/work/tc062/tc062/sanjayb/voice-search-project:$PYTHONPATH

# Load the required modules 
module load python/3.9.13
module load pytorch/1.13.1
source /work/tc062/tc062/sanjayb/voice-search-project/mhubert_model/hubert_env/bin/activate

LANGUAGE=$1
LAYER=$2
MIN_PHONE_SEQ_LENGTH=$3
MAX_PHONE_SEQ_LENGTH=$4
RESULTS_DIR_FOLDER=$5
MODEL_TYPE=$6
WINDOW_SIZE_MS=$7
STRIDE_MS=$8

srun python -u utils/calculate_results.py $LANGUAGE $LAYER $MIN_PHONE_SEQ_LENGTH $MAX_PHONE_SEQ_LENGTH $RESULTS_DIR_FOLDER $MODEL_TYPE $WINDOW_SIZE_MS $STRIDE_MS