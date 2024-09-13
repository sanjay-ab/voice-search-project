#!/bin/bash

LANGUAGE=odia
LAYER=9
WINDOW_SIZE_MS=none
STRIDE_MS=none
MODEL_TYPE=raw_hubert
USE_QUERIES_CUT_AFTER_EMBEDDING=True

# query_pool_job_id=$(sbatch --parsable mhubert_model/job_scripts/hubert_pool_embeddings_job_script.slurm $WINDOW_SIZE_MS $STRIDE_MS $LAYER True False $LANGUAGE $USE_QUERIES_CUT_AFTER_EMBEDDING)
# doc_pool_job_id=$(sbatch --parsable mhubert_model/job_scripts/hubert_pool_doc_embeddings_job_script.slurm $WINDOW_SIZE_MS $STRIDE_MS $LAYER False True $LANGUAGE $USE_QUERIES_CUT_AFTER_EMBEDDING)

# query_preprocess_job_id=$(sbatch --parsable --dependency=afterok:$query_pool_job_id mhubert_model/job_scripts/hubert_ranking_preprocessing_job_script.slurm $WINDOW_SIZE_MS $STRIDE_MS $LAYER True False $LANGUAGE $USE_QUERIES_CUT_AFTER_EMBEDDING)
# doc_preprocess_job_id=$(sbatch --parsable --dependency=afterok:$doc_pool_job_id mhubert_model/job_scripts/hubert_ranking_preprocessing_job_script.slurm $WINDOW_SIZE_MS $STRIDE_MS $LAYER False True $LANGUAGE $USE_QUERIES_CUT_AFTER_EMBEDDING)
# query_preprocess_job_id=$(sbatch --parsable mhubert_model/job_scripts/hubert_ranking_preprocessing_job_script.slurm $WINDOW_SIZE_MS $STRIDE_MS $LAYER True False $LANGUAGE $USE_QUERIES_CUT_AFTER_EMBEDDING)
# doc_preprocess_job_id=$(sbatch --parsable mhubert_model/job_scripts/hubert_ranking_preprocessing_job_script.slurm $WINDOW_SIZE_MS $STRIDE_MS $LAYER False True $LANGUAGE $USE_QUERIES_CUT_AFTER_EMBEDDING)

# ranking_job_id=$(sbatch --parsable --dependency=afterok:$query_preprocess_job_id --dependency=afterok:$doc_preprocess_job_id mhubert_model/job_scripts/hubert_ranking_vectorised_job_script_gpu.slurm $WINDOW_SIZE_MS $STRIDE_MS $LAYER $LANGUAGE $USE_QUERIES_CUT_AFTER_EMBEDDING)

# sbatch --dependency=afterok:$ranking_job_id utils/job_scripts/calculate_results_extra_args.slurm $LANGUAGE $LAYER 0 0 "None" $MODEL_TYPE $WINDOW_SIZE_MS $STRIDE_MS
sbatch utils/job_scripts/calculate_results_extra_args.slurm $LANGUAGE $LAYER 0 0 "None" $MODEL_TYPE $WINDOW_SIZE_MS $STRIDE_MS