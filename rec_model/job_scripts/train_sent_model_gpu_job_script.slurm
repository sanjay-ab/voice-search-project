#!/bin/bash
#
#SBATCH --job-name=train_rec_model
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=tc062-pool3
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

export HF_HOME=/work/tc062/tc062/sanjayb/.cache/hugging_face_cache
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export PYTHONPATH=/work/tc062/tc062/sanjayb/dissertation:$PYTHONPATH

# Load the required modules 
module load python/3.10.8-gpu
module load pytorch/1.13.1-gpu
source /work/tc062/tc062/sanjayb/dissertation/mhubert_model/hubert_env_gpu/bin/activate

TEMPERATURE=0.15
LR=0.0001
OUTPUT_DIM=8192
NO_GRAD_ON_AWE_MODEL=False
MODEL_TYPE=standard

srun python -u rec_model/train_model.py $TEMPERATURE $LR $OUTPUT_DIM $NO_GRAD_ON_AWE_MODEL $MODEL_TYPE