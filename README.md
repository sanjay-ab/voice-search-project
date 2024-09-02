# Voice Search Project

## Overall Description
This project explores different methods of producing a voice search system for low-resource unwritten languages, using transfer learning.
The search task is formulated as a spoken Query-by-Example task.
Models are trained/tested on Tamil and then tested directly on Gormati, an unwritten Indic language, without finetuning.
Each system uses different speech representations and uses cosine similarity matching to match queries and documents.
This code was written as part of an MSc dissertation.
All work is my own, except that found in the folders: voice_search_server and speechseqembeddings, which are slightly modified from their originals:  [1] and [2], respectively.

## Description of each system
Each system converts queries and documents into vector representations and compares them using a method based on cosine similarity.
This section describes the representations used by each of the systems.
### mHuBERT System
This system converts queries and documents into mHuBERT vectors using a pretrained mHuBERT model [3].

### Acoustic Word Embedding (AWE) System
This system uses a learned pooling model (adapted from [2]) which can take a segment of speech and convert it into a single vector embedding.
This model was trained on segments of Tamil speech. It was trained contrastively, to embed segments with the same phonetic transcription similarly and segments with different transcriptions differently.
Whole recordings (queries and documents) were embedded using a sliding window approach - a window was slid across the recording and each window was embedded.
This collection of embeddings is the AWE representation of the recording.
Note that windows can be specified either in time (e.g., 240 ms) or in phones (e.g., 3 phones) if the recordings have phone transcriptions.

### Recording Embedding System
This system uses a learned pooling layer adapted from the AWE model to take a recording and convert it into a single vector embedding. 
This model was trained contrastively on Tamil, to embed recordings containing the same keyword similarly and recordings containing different keywords differently. 

## Data
The Tamil data was used for debugging and training the models since it had existing gold transcriptions and was higher resource than the target unwritten Gormati data.
The Tamil data was taken from the Indic ASR challenge dataset [4] and converted into a keyword search task for our project. 

The target data was collected from a Banjara farming community in India, who speak Gormati, a language with no written form [5]. 
This data is very low-resource (around 300 files and <4 hours of data), so it was only used for testing, not training.
The data was already formatted into classes, with each recording being a natural description of its respective class label.
To convert this into a search task, queries were simply randomly selected from each class.

## References
[1] https://github.com/unmute-tech/voice-search-server

[2] https://gitlab.cognitive-ml.fr/ralgayres/speechseqembeddings

[3] https://huggingface.co/utter-project/mHuBERT-147

[4] Diwan, A., Vaideeswaran, R., Shah, S., Singh, A., Raghavan, S., Khare, S., Unni, V., Vyas, S., Rajpuria, A., Yarra, C., Mittal, A., Ghosh, P.K., Jyothi, P., Bali, K., Seshadri, V., Sitaram, S., Bharadwaj, S., Nanavati, J., Nanavati, R., Sankaranarayanan, K. (2021) MUCS 2021: Multilingual and Code-Switching ASR Challenges for Low Resource Indian Languages. Proc. Interspeech 2021, 2446-2450, doi: 10.21437/Interspeech.2021-1339

[5] Reitmaier, T, Kalarikalayil Raju, D, Klejch, O, Wallington, E, Markl, N, Pearson, J, Jones, M, Bell, P & Robinson, S 2024, Cultivating Spoken Language Technologies for Unwritten Languages. in CHI '24: Proceedings of the 2024 CHI Conference on Human Factors in Computing Systems. ACM, pp. 1-17, ACM CHI Conference on Human Factors in Computing Systems 2024, Honolulu, Hawaii, United States, 11/05/24.
