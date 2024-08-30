#!/bin/bash

TEMPERATURE=0.07
# LR=0.0001
LR=1e-05
MIDDLE_DIM=512
OUTPUT_DIM=2048
NO_GRAD_STRING="grad"

LANGUAGE=banjara
LAYER=9
EPOCH=3
# SAVE_EMBEDDING_FOLDER="finetune_awe_${NO_GRAD_STRING}_1_layer_middle_dim_512_output_dim_${OUTPUT_DIM}_lr_${LR}_tmp_${TEMPERATURE}_weight_decay_0.0_${EPOCH}"
SAVE_EMBEDDING_FOLDER="finetune_awe_lr_${LR}_tmp_${TEMPERATURE}_${EPOCH}"
# MODEL_SAVE_DIR="data/tamil/models/sent/${LAYER}/finetune_awe_${NO_GRAD_STRING}_1_layer_middle_dim_512_output_dim_${OUTPUT_DIM}_lr_${LR}_tmp_${TEMPERATURE}_weight_decay_0.0"
MODEL_SAVE_DIR="data/tamil/models/sent/${LAYER}/finetune_awe_grad_lr_${LR}_tmp_${TEMPERATURE}"
MODEL_NAME=2024-08-06_22:19:10_checkpoint_epoch_${EPOCH}.pt
# RESULTS_PATH="${LAYER}/finetune_awe_${NO_GRAD_STRING}_1_layer_middle_dim_512_output_dim_${OUTPUT_DIM}_lr_${LR}_tmp_${TEMPERATURE}_weight_decay_0.0_${EPOCH}"
RESULTS_PATH="${LAYER}/finetune_awe_grad_lr_${LR}_tmp_${TEMPERATURE}_${EPOCH}"
MODEL_TYPE=sent
USE_AWES=False

query_sent_job_id=$(sbatch --parsable rec_model/job_scripts/extract_query_sent_embeddings_extra_args.slurm $LANGUAGE $LAYER $SAVE_EMBEDDING_FOLDER $MODEL_SAVE_DIR $MODEL_NAME $USE_AWES $MIDDLE_DIM $OUTPUT_DIM)
doc_sent_job_id=$(sbatch --parsable rec_model/job_scripts/extract_doc_sent_embeddings_parallel_extra_args.slurm $LANGUAGE $LAYER $SAVE_EMBEDDING_FOLDER $MODEL_SAVE_DIR $MODEL_NAME $USE_AWES $MIDDLE_DIM $OUTPUT_DIM)

ranking_job_id=$(sbatch --parsable --dependency=afterok:$query_sent_job_id --dependency=afterok:$doc_sent_job_id rec_model/job_scripts/ranking_vectorised_extra_args.slurm $LANGUAGE $LAYER $SAVE_EMBEDDING_FOLDER $RESULTS_PATH)

sbatch --dependency=afterok:$ranking_job_id utils/job_scripts/calculate_results_extra_args.slurm $LANGUAGE $LAYER 0 0 $RESULTS_PATH $MODEL_TYPE