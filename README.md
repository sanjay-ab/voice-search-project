# Voice Search Project

## Overall Description
This project focuses on different methods of producing a voice search system for low-resource unwritten languages, using transfer learning.
The search task is formulated as a spoken Query-by-Example task.
Models are trained/tested on Tamil and then tested directly on Gormati, an unwritten Indic language, without finetuning.
Each system uses different speech representations and uses cosine similarity matching to match queries and documents.
This code was written as part of an MSc dissertation.
All work is my own, except that found in the folders: voice_search_server and speechseqembeddings, which are slightly modified from their originals: https://github.com/unmute-tech/voice-search-server and https://gitlab.cognitive-ml.fr/ralgayres/speechseqembeddings, respectively.

## Description of each system
Each system converts queries and documents into vector representations and compares them using a method based on cosine similarity.
This section describes the representations used by each of the systems.
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
This model is trained contrastively on Tamil, to embed recordings containing the same keyword similarly and recordings containing different keywords differently. 

## Data
Tamil data was used for debugging and training since it had existing gold transcriptions and was higher resource than the target unwritten Gormati data.
This data was taken from the Indic ASR challenge dataset (https://arxiv.org/pdf/2104.00235). 
Recordings with fewer than 3 words were removed, since they were mainly simple conversational responses. 
The remaining recordings were split into queries and documents.
This split was done by extracting keyword queries.
The keywords were found through tf-idf, they were chosen such that each query had between 1 and 5 relevant documents. 
The documents from which the queries were extracted from were left in the search collection but were ignored in the rankings for the related query.

The target data was collected from a Banjara farming community in India, who speak Gormati, a language with no written form (https://dl.acm.org/doi/pdf/10.1145/3613904.3642026). 
This data was also split into queries and documents.
To do this, first, any classes with only 1 file were removed, then queries were randomly selected from the classes.
Here, the queries are simply whole uncut recordings.
Queries were randomly sampled, so that the number of queries in each class was proportional to the number of documents in that class.
Additionally, it was ensured that each class had at least one query.   
Queries and documents were left in the search collection and a match of a query with itself was ignored during ranking.
This data was very low-resource (around 300 files and <4 hours of data), so it was only used for testing, not training.