#!/bin/bash
#
#SBATCH --job-name=hubert_doc_ranking_preprocessing
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --account=tc062-pool3
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=96g

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

srun python -u mhubert_model/ranking_preprocessing.py $WINDOW_SIZE_MS $STRIDE_MS $LAYER $RUN_FOR_QUERIES $RUN_FOR_DOCUMENTS