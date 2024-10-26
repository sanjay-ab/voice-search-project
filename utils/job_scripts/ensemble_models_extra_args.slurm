#!/bin/bash
#
#SBATCH --job-name=ensemble_models  
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

TOP_FOLDER_NAME_SUFFIX=$1
MODEL_LANGUAGES=$2
RESULTS_DIR_FOLDER=$3
MODEL_TYPE=$4
LAYER=$5
TEST_LANGUAGE=$6

srun python -u utils/ensemble_model_results.py $TOP_FOLDER_NAME_SUFFIX $MODEL_LANGUAGES $RESULTS_DIR_FOLDER $MODEL_TYPE $LAYER $TEST_LANGUAGE