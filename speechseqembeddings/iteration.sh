#!/bin/sh

# Trains a SSE model on positive pairs found with kNN search. Early stopping is done with MAP. To do early stopping on test loss uncomment lines 18,19,20.

pretrained_model=$1
output_dir=$2
#features='features/train-clean-360-subset' 
#train_item='training_data/train-clean-360-subset-segments'
features='features/dev-clean' 
train_item='training_data/dev-clean-segments'

# training is done with early stopping on MAP values
map_features='features/dev-clean/' 
map_segments='sse_benchmark/dev-clean-ngrams-subset'
valid_on_map='--valid_on_map'

# training is done with early stopping on train loss applied to test set.
#test_features='features/dev-clean/' 
#test_item='training_data/dev-clean-segments'
#valid_on_map=''

threshold=0.5
mode='pairs'
n_cpus=40
knn_k=10 #nb of neighbors in knn
temp=0.15
epoch_size=1000
max_t=100
max_patience=1
lr=0.0001
mkdir -p $output_dir
pair_file=$output_dir/pairs_50percent_segments
test_pair_file=$output_dir/test_pairs_50percent_segments



# KNN SEARCH
if [ -f "$pair_file" ]; then
    echo $pair_file already exist
else
    echo $output_dir
    # searching for positive pairs
    python utils/knn/batch_main_rnn.py $output_dir/train/ $features $n_cpus $pretrained_model $train_item $knn_k
    # selecting positive pairs
    cat $output_dir/train/pairs_folder/* > $output_dir/train/pairs
    nb_segments=$(cat $train_item | wc -l)
    python utils/knn/get_threshold_pairs.py $output_dir/train/pairs $nb_segments $pair_file $threshold
    echo $pair_file
fi

# TEST KNN SEARCH
if [ -f "$test_pair_file" ]; then
    echo $test_pair_file already exist
#else
elif [ "$valid_on_map" != '--valid_on_map' ]; then
    echo $output_dir
    # searching for positive pairs
    python utils/knn/batch_main_rnn.py $output_dir/test/ $test_features $n_cpus $pretrained_model $test_item $knn_k
    # selecting positive pairs
    cat $output_dir/test/pairs_folder/* > $output_dir/test/pairs
    nb_segments=$(cat $test_item | wc -l)
    python utils/knn/get_threshold_pairs.py $output_dir/test/pairs $nb_segments $test_pair_file $threshold
    echo $test_pair_file
fi
mkdir -p $output_dir/checkpoints/

python utils/train_sse.py --mode=$mode --path_train_item=$pair_file --path_test_item=$test_pair_file --path_map_item=$map_segments --path_features=$features --path_test_features=$test_features --path_map_features=$map_features --output_dir=$output_dir/checkpoints/ --temperature=$temp --max_t=$max_t $valid_on_map --epoch_size=$epoch_size --max_patience=$max_patience --learning_rate=$lr

echo trained model is saved at $output_dir/checkpoints/ 
