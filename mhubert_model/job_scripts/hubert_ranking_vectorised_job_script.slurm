#!/bin/bash
#
#SBATCH --job-name=hubert_doc_ranking_vectorised
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --account=tc062-pool3
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --array=7-11
#SBATCH --mem=64g

echo $SLURM_ARRAY_TASK_ID

export HF_HOME=/work/tc062/tc062/sanjayb/.cache/hugging_face_cache
export PYTHONPATH=/work/tc062/tc062/sanjayb/dissertation:$PYTHONPATH

# Load the required modules 
module load python/3.9.13
module load pytorch/1.13.1
source /work/tc062/tc062/sanjayb/dissertation/mhubert_model/hubert_env/bin/activate

srun python mhubert_model/query_document_search_vectorised.py $SLURM_ARRAY_TASK_ID