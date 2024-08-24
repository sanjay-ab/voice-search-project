#!/bin/bash

LANGUAGE=banjara
LAYER=9
SAVE_EMBEDDING_FOLDER="hubert_lr_0.001_linear_weight_decay_0.1"
MODEL_SAVE_DIR="data/tamil/models/sent/${LAYER}/hubert_lr_0.001_linear_weight_decay_0.1"
MODEL_NAME=2024-07-26_11:40:59_checkpoint_epoch_10.pt
RESULTS_PATH="${LAYER}/hubert_lr_0.001_linear_weight_decay_0.1"
MODEL_TYPE=sent
USE_AWES=False

query_sent_job_id=$(sbatch --parsable sent_model/job_scripts/extract_query_sent_embeddings_extra_args.slurm $LANGUAGE $LAYER $SAVE_EMBEDDING_FOLDER $MODEL_SAVE_DIR $MODEL_NAME $USE_AWES)
doc_sent_job_id=$(sbatch --parsable sent_model/job_scripts/extract_doc_sent_embeddings_parallel_extra_args.slurm $LANGUAGE $LAYER $SAVE_EMBEDDING_FOLDER $MODEL_SAVE_DIR $MODEL_NAME $USE_AWES)

ranking_job_id=$(sbatch --parsable --dependency=afterok:$query_sent_job_id --dependency=afterok:$doc_sent_job_id sent_model/job_scripts/ranking_vectorised_extra_args.slurm $LANGUAGE $LAYER $SAVE_EMBEDDING_FOLDER $RESULTS_PATH)

sbatch --dependency=afterok:$ranking_job_id utils/job_scripts/calculate_results_extra_args.slurm $LANGUAGE $LAYER 0 0 $RESULTS_PATH $MODEL_TYPE