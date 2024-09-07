#!/bin/bash

LANGUAGE=banjara
PHONE_TIMINGS_FNAME=phone_all_mpr.ctm
MIN_PHONE_SEQ_LENGTH=5
MAX_PHONE_SEQ_LENGTH=14
USE_WINDOW_SPLITTER_INSTEAD_OF_PHONE_SPLITTER=True
LAYER=9
LR=1e-4
TMP=0.07
ACC=1000
BS=5
EPOCH=0
PREFIX="half_"
TRAIN_SEQ=3_9
SAVE_EMBEDDING_FOLDER="${PREFIX}lr_${LR}_tmp_${TMP}_acc_${ACC}_bs_${BS}_epoch_${EPOCH}_${MIN_PHONE_SEQ_LENGTH}_${MAX_PHONE_SEQ_LENGTH}_${TRAIN_SEQ}"
MODEL_SAVE_DIR="data/tamil/models/awe/${LAYER}/${PREFIX}lr_${LR}_tmp_${TMP}_acc_${ACC}_bs_${BS}_${TRAIN_SEQ}"
MODEL_NAME="2024-07-23_12:51:17_checkpoint_epoch_${EPOCH}.pt"
RESULTS_PATH="${LAYER}/tamil_train_3_9/${PREFIX}lr_${LR}_tmp_${TMP}_acc_${ACC}_bs_${BS}_${MIN_PHONE_SEQ_LENGTH}_${MAX_PHONE_SEQ_LENGTH}_${TRAIN_SEQ}_epoch_${EPOCH}"
MODEL_TYPE="awe"

query_awe_job_id=$(sbatch --parsable awe_model/job_scripts/extract_query_awe_embeddings_extra_args.slurm $LANGUAGE $PHONE_TIMINGS_FNAME $MIN_PHONE_SEQ_LENGTH $MAX_PHONE_SEQ_LENGTH $USE_WINDOW_SPLITTER_INSTEAD_OF_PHONE_SPLITTER $LAYER $SAVE_EMBEDDING_FOLDER $MODEL_SAVE_DIR $MODEL_NAME)
doc_awe_job_id=$(sbatch --parsable awe_model/job_scripts/extract_doc_awe_embeddings_parallel_extra_args.slurm $LANGUAGE $PHONE_TIMINGS_FNAME $MIN_PHONE_SEQ_LENGTH $MAX_PHONE_SEQ_LENGTH $USE_WINDOW_SPLITTER_INSTEAD_OF_PHONE_SPLITTER $LAYER $SAVE_EMBEDDING_FOLDER $MODEL_SAVE_DIR $MODEL_NAME)

ranking_job_id=$(sbatch --parsable --dependency=afterok:$query_awe_job_id --dependency=afterok:$doc_awe_job_id awe_model/job_scripts/ranking_vectorised_extra_args.slurm $LANGUAGE $LAYER $MIN_PHONE_SEQ_LENGTH $MAX_PHONE_SEQ_LENGTH $SAVE_EMBEDDING_FOLDER $RESULTS_PATH)

sbatch --dependency=afterok:$ranking_job_id utils/job_scripts/calculate_results_extra_args.slurm $LANGUAGE $LAYER $MIN_PHONE_SEQ_LENGTH $MAX_PHONE_SEQ_LENGTH $RESULTS_PATH $MODEL_TYPE