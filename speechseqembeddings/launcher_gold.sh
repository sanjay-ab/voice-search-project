#!/bin/sh
# Train a model with contrastive learning using pairs of matching sequence of phonemes. 

OUTPUT_DIR='checkpoints_gold/'
TRAIN_FEATURES='features/train-clean-360/'
TRAIN_SEGMENTS='training_data/train-clean-360-subset-ngrams'
MAP_SEGMENTS='sse_benchmark/dev-clean-ngrams-subset'
MAP_FEATURES='features/dev-clean/'


mkdir -p $OUTPUT_DIR
MODE='gold'
epoch=60
temp=0.15
epoch_size=1000
valid_on_map='--valid_on_map'
lr=0.0001

python utils/train_sse.py --mode=$MODE --path_train_item=$TRAIN_SEGMENTS --path_features=$TRAIN_FEATURES --path_map_item=$MAP_SEGMENTS --path_map_features=$MAP_FEATURES --output_dir=$OUTPUT_DIR --max_t=$epoch --temperature=$temp --epoch_size=$epoch_size $valid_on_map --learning_rate=$lr
