# Speech Sequence Embeddings using Nearest Neighbors Contrastive Learning
This code can be used to reproduce the main results from [our paper](https://arxiv.org/abs/2204.05148) 

from Robin Algayres, Adel Nabli, Benoit Sagot, Emmanuel Dupoux

[Our website](https://cognitive-ml.fr/)

### Installation:

Install this specific fairseq package:

```git clone https://gitlab.cognitive-ml.fr/ralgayres/fairseq_fork```

And follow the installation procedure described in fairseq_fork/README.md


### This github contains:

- the pretrained Wav2vec2 model (small version) and pretrained SSE models (unsupervised and weakly supervised version)
- training files to reproduce our results on a small subset the LibriSpeech corpus
- note that using more than 10 hours of speech data for our SSE training method does not enable to reach higher performances.

### Training of SSE model on LibriSpeech:

Segment your the corpus into VAD sections (voice activity detection) and format each VAD as follows (e.g: training_data/train-clean-360-subset-vads): 

\<path to wav\> \<spearker id\> \<vad start\> \<vad end\> 

We advise using pyannote to get the VAD sections. Then, get the speech segments that will be used for making positive pairs with the following script:

```python utils/make_segments.py training_data/train-clean-360-subset-vads \<outputfile\>```

Each line is formatted as follows:

\<path to wav\> \<spearker id\> \<vad start\> \<vad end\> \<segment start\> \<segment end\> \<transcription (optional)\>

The latter outputfile has already been computed and can be found at training_data/train-clean-360-subset-segments

Get a path to the LibriSpeech corpus:
    ```LS=\<path to LibriSpeech folder\>```

Log on a machine with one GPU with 16Go of RAM and if possible 20 or more cpus. Then launch the following commands.

Extract Wav2vec2.0 frames for training and test set:

    ```python utils/extract_features.py --path_wavs=$LS --path_vads=training_data/train-clean-360-subset-vads --output_dir=features/train-clean-360-subset/```

    ```python utils/extract_features.py --path_wavs=$LS --path_vads=training_data/dev-clean-vads --output_dir=features/dev-clean-subset```

Train the initial SSE model with data augmentation 

    ```bash launch_init.sh $LS```

Extract k-NN positive pairs and train a new SSE model

    ```bash iteration.sh $LS checkpoints/\<model trained with previous script\>``` 

The previous script can be run again once the new SSE model is trained until the MAP scores stops increasing (or test loss stops decreasing).

Optional: train a weakly supervised model, you will need to transcribe phonetically the speech segments created by utils/make_segments.py: 

    ```git clone https://gitlab.cognitive-ml.fr/ralgayres/miscellaneuous/```

    ```python utils/transcribe_timestamps.py training_data/train-clean-360-subset-segments miscellaneous/ls_alignments/phones/train-clean-360/ outputfile```

The latter outputfile has already been computed and can be found at training_data/train-clean-360-subset-ngrams.

Launch the weakly supervised training:

    ```bash launch_gold.sh $LS```


#### Evaluating and creating learned embeddings

Get a list of transcribed speech segments (e.g sse_benchmark/dev-clean-ngrams-subset) where each line is formatted as follows:

\<path to wav\> \<spearker id\> \<vad start\> \<vad end\> \<segment start\> \<segment end\> \<transcription (optional, if you wish to compute MAP)\>

If you do not have vads' timecodes, you can simply copy the segments' timecodes instead.

Using a pretrained SSE model, you can compute a MAP score or embed the segments into a file.

To get an MAP score: 

    ```python utils/map_dump.py --path_wavs=$LS --path_segments=sse_benchmark/dev-clean-ngrams-subset --task=map --path_sse=pretrained/librispeech_unsup/```

To embed a list of segments: 

    ```python utils/map_dump.py --path_wavs=$LS --path_segments=sse_benchmark/dev-clean-ngrams-subset --task=dump --path_sse=pretrained/librispeech_unsup/ --output_file=out```

By default, models are trained until the MAP score computed on the validation set reaches the highest value. If instead you prefer to use validation on the training loss, you can uncomment the line "valid_on_map=\'\'" in training script files.
