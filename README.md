# Voice Search Project

## Overall Description
Search systems are widely used in modern digital technology, but they are primarily text-based. 
This creates challenges for communities that speak languages without a written form, limiting their access to these technologies. 
While speech-based search systems could offer a valuable alternative, the limited resources available for unwritten languages mean that they are rarely supported.
To address this issue, this project explores different methods of producing a voice search system for low-resource unwritten languages, using transfer learning.
The search task is formulated as a spoken Query-by-Example task.
Models are trained/tested on various indic languages, including Tamil, Telugu and Gujarati, and then tested directly on Gormati, an unwritten Indic language, without finetuning.
Each system uses different speech representations and uses cosine similarity matching to match queries and documents.
All work is my own, except that found in the folders: voice_search_server and speechseqembeddings, which are slightly modified from their originals:  [[1]](#1) and [[2]](#2), respectively.

This work was published at EMNLP 2025 here: https://aclanthology.org/2025.findings-emnlp.1224/

Citation: Sanjay Booshanam, Kelly Chen, Ondrej Klejch, Thomas Reitmaier, Dani Kalarikalayil Raju, Electra Wallington, Nina Markl, Jennifer Pearson, Matt Jones, Simon Robinson, and Peter Bell. 2025. Spoken Document Retrieval for an Unwritten Language: A Case Study on Gormati. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 22497–22509, Suzhou, China. Association for Computational Linguistics.

## Description of each system
Each system converts queries and documents into vector representations and compares them using a method based on cosine similarity.
This section describes the representations used by each of the systems.
### mHuBERT System
This system converts queries and documents into mHuBERT vectors using a pretrained mHuBERT model [[3]](#3).

### Acoustic Word Embedding (AWE) System
This system uses a learned pooling model (adapted from [[2]](#2)) which can take a segment of speech and convert it into a single vector embedding.
Models were trained on segments of Tamil, Telugu and Gujarati speech. They were trained contrastively, to embed segments with the same phonetic transcription similarly and segments with different transcriptions differently.
Whole recordings (queries and documents) were embedded using a sliding window approach - a window was slid across the recording and each window was embedded.
This collection of embeddings is the AWE representation of the recording.
Note that windows can be specified either in time (e.g., 240 ms) or in phones (e.g., 3 phones) if the recordings have phone transcriptions.

### Recording Embedding System
This system uses a learned pooling layer adapted from the AWE model to take a recording and convert it into a single vector embedding. 
This model was trained contrastively on Tamil, to embed recordings containing the same keyword similarly and recordings containing different keywords differently.

## Data
Indic language data from the Indic ASR challenge dataset [[4]](#4) was converted into a keyword search task for our project and used for debugging and training the models since they had existing gold transcriptions and were higher resource than the target unwritten Gormati data.

The target data was collected from a Banjara farming community in India, who speak Gormati, a language with no written form [[5]](#5).
This data is very low-resource (around 300 files and <4 hours of data), so it was only used for testing, not training.
The data was already formatted into classes, with each recording being a natural description of its respective class label.
To convert this into a search task, queries were simply randomly selected from each class.

## References
<a id="1">[1]</a>
https://github.com/unmute-tech/voice-search-server

<a id="2">[2]</a>
https://gitlab.cognitive-ml.fr/ralgayres/speechseqembeddings

<a id="3">[3]</a>
https://huggingface.co/utter-project/mHuBERT-147

<a id="4">[4]</a>
Diwan, A., Vaideeswaran, R., Shah, S., Singh, A., Raghavan, S., Khare, S., Unni, V., Vyas, S., Rajpuria, A., Yarra, C., Mittal, A., Ghosh, P.K., Jyothi, P., Bali, K., Seshadri, V., Sitaram, S., Bharadwaj, S., Nanavati, J., Nanavati, R., Sankaranarayanan, K. (2021) MUCS 2021: Multilingual and Code-Switching ASR Challenges for Low Resource Indian Languages. Proc. Interspeech 2021, 2446-2450, doi: 10.21437/Interspeech.2021-1339

<a id="5">[5]</a>
Reitmaier, T, Kalarikalayil Raju, D, Klejch, O, Wallington, E, Markl, N, Pearson, J, Jones, M, Bell, P & Robinson, S 2024, Cultivating Spoken Language Technologies for Unwritten Languages. in CHI '24: Proceedings of the 2024 CHI Conference on Human Factors in Computing Systems. ACM, pp. 1-17, ACM CHI Conference on Human Factors in Computing Systems 2024, Honolulu, Hawaii, United States, 11/05/24.
