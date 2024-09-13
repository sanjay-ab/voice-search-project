#!/bin/bash
#
#SBATCH --job-name=extract_phone_timings
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --account=tc062-pool3
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --array=0-63

echo $SLURM_ARRAY_TASK_ID

export HF_HOME=/work/tc062/tc062/sanjayb/.cache/hugging_face_cache
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export PYTHONPATH=/work/tc062/tc062/sanjayb/voice-search-project:$PYTHONPATH

# Load the required modules 
module load python/3.7.16
export CONDA_ROOT=/work/tc062/tc062/sanjayb/.condaenvs
export CONDARC=${CONDA_ROOT}/.condarc
eval "$(conda shell.bash hook)"
conda activate voice-search-server

LANGUAGE=odia
N_PARTS=64

srun python -u utils/extract_phone_timings.py $N_PARTS $SLURM_ARRAY_TASK_ID $LANGUAGE