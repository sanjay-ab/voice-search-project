#!/bin/bash

if [ $# -ne 5 ]; then
    cat "config file + exp name + corpus"
    exit
fi
output_dir=$1 
feature_dir=$2
n_cpus=$3
pretrained_model=$4
segments=$5

nb_segments=$(cat $segments | wc -l)
echo 'nb of segments in' $corpus 'is' $nb_segments
echo $output_dir $config_file
python utils/knn/batch_main_rnn.py $output_dir $feature_dir $n_cpus $pretrained_model $segments
cat $output_dir/pairs_folder/* > $output_dir/pairs
python utils/knn/get_threshold_pairs.py $output_dir/pairs $nb_segments $output_dir/pairs_50percent_segments

#rm -r $output_dir/embeddings/ $output_dir/faiss/
#rm -r $output_dir/pairs_folder $output_dir/clusters_folder 
#rm $output_dir/pairs

echo "done"
