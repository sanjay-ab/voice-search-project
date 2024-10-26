#!/bin/bash

LANGUAGE=banjara
PHONE_TIMINGS_FNAME=phone_all.ctm
MIN_PHONE_SEQ_LENGTH=5
MAX_PHONE_SEQ_LENGTH=14
WINDOW_STR="win_"
USE_QUERIES_CUT_AFTER_EMBEDDING=True
LAYER=9
LR=0.0001
TMP=0.07
ACC=1000
BS=5
MODEL_LANGUAGES="gujarati_telugu_tamil"
EPOCH=best
PREFIX=""
TRAIN_SEQ=3_9

TOP_FOLDER_NAME_SUFFIX="_train_3_9_queries_cut_after_embedding"
RESULTS_FOLDER="${WINDOW_STR}${PREFIX}lr_${LR}_tmp_${TMP}_acc_${ACC}_bs_${BS}_${MIN_PHONE_SEQ_LENGTH}_${MAX_PHONE_SEQ_LENGTH}_${TRAIN_SEQ}_epoch_${EPOCH}"
RESULTS_PATH="${LAYER}/ensemble_${MODEL_LANGUAGES}${TOP_FOLDER_NAME_SUFFIX}/${RESULTS_FOLDER}"
MODEL_TYPE="awe"


ensemble_job_id=$(sbatch --parsable utils/job_scripts/ensemble_models_extra_args.slurm $TOP_FOLDER_NAME_SUFFIX $MODEL_LANGUAGES $RESULTS_FOLDER $MODEL_TYPE $LAYER $LANGUAGE)

sbatch --dependency=afterok:$ensemble_job_id utils/job_scripts/calculate_results_extra_args.slurm $LANGUAGE $LAYER $MIN_PHONE_SEQ_LENGTH $MAX_PHONE_SEQ_LENGTH $RESULTS_PATH $MODEL_TYPE