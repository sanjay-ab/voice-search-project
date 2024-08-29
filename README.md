# Voice Search Project

## Overall Description
This project focuses on different methods of producing a voice search system for low-resource unwritten languages, using transfer learning.
The search task is formulated as a spoken Query-by-Example task.
Models are trained/tested on Tamil and then tested directly on Gormati, an unwritten Indic language, without finetuning.
Each system uses different speech representations and uses cosine similarity matching to match queries and documents.
This code was written as part of an MSc dissertation.
All work is my own, except that found in the folders: voice_search_server and speechseqembeddings, which are slightly modified from their originals: https://github.com/unmute-tech/voice-search-server and https://gitlab.cognitive-ml.fr/ralgayres/speechseqembeddings, respectively.

## Description of each system
### mHuBERT System
This system converts queries and documents into mHuBERT vectors using a pretrained mHuBERT model (https://huggingface.co/utter-project/mHuBERT-147).

### Acoustic Word Embedding (AWE) System
This system uses a learned pooling model (adapted from https://gitlab.cognitive-ml.fr/ralgayres/speechseqembeddings) which can take a segment of speech and convert it into a single vector embedding.
This model is trained on segments of Tamil speech. It is trained contrastively, to embed segments with the same phonetic transcription similarly and segments with different transcriptions differently.
Whole recordings (queries and documents) are embedded using a sliding window approach - a window is slid across the recording and each window is embedded.
This collection of embeddings is the AWE representation of the recording.
Note that windows can be specified either in time (e.g., 240 ms) or in phones (e.g., 3 phones), if the recordings have phone transcriptions.

### Recording Embedding System
This system uses a learned pooling layer adapted from the AWE model to take a recording and convert it into a single vector embedding. 
This model is trained contrastively on Tamil, to embed recordings containing the same keyword similarly and different keywords differently. 