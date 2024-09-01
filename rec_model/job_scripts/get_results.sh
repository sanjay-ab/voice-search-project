#!/bin/bash

TEMPERATURE=0.07
# LR=0.0001
LR=1e-05
OUTPUT_DIM=2048
NO_GRAD_STRING="grad"
MODEL_TYPE=standard  # standard or extra_linear 

if [ "${MODEL_TYPE}" = "standard" ]; then
    CENTER_STRING=""
else
    CENTER_STRING="1_layer_output_dim_${OUTPUT_DIM}_"
fi

LANGUAGE=banjara
LAYER=9
EPOCH=3
SAVE_EMBEDDING_FOLDER="finetune_awe_${NO_GRAD_STRING}_${CENTER_SAVE_STRING}lr_${LR}_tmp_${TEMPERATURE}_${EPOCH}"
MODEL_SAVE_DIR="data/tamil/models/rec/${LAYER}/finetune_awe_${NO_GRAD_STRING}_${CENTER_SAVE_STRING}lr_${LR}_tmp_${TEMPERATURE}"
MODEL_NAME=2024-08-06_22:19:10_checkpoint_epoch_${EPOCH}.pt
RESULTS_PATH="${LAYER}/finetune_awe_${NO_GRAD_STRING}_${CENTER_SAVE_STRING}lr_${LR}_tmp_${TEMPERATURE}_${EPOCH}"
MODEL_TYPE=rec

query_rec_job_id=$(sbatch --parsable rec_model/job_scripts/extract_query_rec_embeddings_extra_args.slurm $LANGUAGE $LAYER $SAVE_EMBEDDING_FOLDER $MODEL_SAVE_DIR $MODEL_NAME $OUTPUT_DIM $MODEL_TYPE)
doc_rec_job_id=$(sbatch --parsable rec_model/job_scripts/extract_doc_rec_embeddings_parallel_extra_args.slurm $LANGUAGE $LAYER $SAVE_EMBEDDING_FOLDER $MODEL_SAVE_DIR $MODEL_NAME $OUTPUT_DIM $MODEL_TYPE)

ranking_job_id=$(sbatch --parsable --dependency=afterok:$query_rec_job_id --dependency=afterok:$doc_rec_job_id rec_model/job_scripts/ranking_vectorised_extra_args.slurm $LANGUAGE $LAYER $SAVE_EMBEDDING_FOLDER $RESULTS_PATH)

sbatch --dependency=afterok:$ranking_job_id utils/job_scripts/calculate_results_extra_args.slurm $LANGUAGE $LAYER 0 0 $RESULTS_PATH $MODEL_TYPE