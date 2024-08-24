#!/bin/sh

#Given a VAD file, a SSE model is trained using data augmentation on the VAD segments. MAP metric is used for early stopping. Uncomment line 15 and 16 to use the training loss on test set for early stopping (i.e. full unsupervised setting). In both cases, training is done with patience=2 (keep training until two consecutive epochs do not achieve lower MAP score/test loss).

WAVS_PATH=$1
OUTPUT_DIR='checkpoints_init/'
VAD_FILE='training_data/train-clean-360-subset-vads'

# train using MAP score for early stopping

MAP_FEATURES='features/dev-clean/'
MAP_SEGMENTS='sse_benchmark/dev-clean-ngrams-subset'  # to compute MAP scores (both can be used)
valid_on_map='--valid_on_map'

# train using training loss on test set for early stopping

#TEST_SEGMENTS='training_data/dev-clean-vads' # to compute training loss of a test set
#valid_on_map=''

mkdir -p $OUTPUT_DIR
epoch_size=200
MODE='vad_aug'
max_t=75
max_patience=2
learning_rate=0.0001
temp=0.15

python utils/train_sse.py --mode=$MODE --path_train_item=$VAD_FILE --path_test_item=$TEST_SEGMENTS --path_map_features=$MAP_FEATURES --path_map_item=$MAP_SEGMENTS --path_wavs=$WAVS_PATH --output_dir=$OUTPUT_DIR --epoch_size=$epoch_size $valid_on_map --max_t=$max_t --max_patience=$max_patience --learning_rate=$learning_rate --temperature=$temp
