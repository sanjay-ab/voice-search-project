#!/bin/bash
#
#SBATCH --job-name=hubert_doc_ranking_vectorised
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --account=tc062-pool3
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=96g

export HF_HOME=/work/tc062/tc062/sanjayb/.cache/hugging_face_cache
export PYTHONPATH=/work/tc062/tc062/sanjayb/voice-search-project:$PYTHONPATH

# Load the required modules 
module load python/3.9.13
module load pytorch/1.13.1
source /work/tc062/tc062/sanjayb/voice-search-project/mhubert_model/hubert_env/bin/activate

srun python -u awe_model/query_document_search_vectorised.py 