import os
import sys
import pickle as pkl
import torch
import torch.nn.functional as F
import multiprocessing as mp
from mhubert_model.collate_embeddings import collate_embeddings
from utils.common_functions import parse_boolean_input, split_list_into_n_parts_and_get_part, make_dir

SAMPLE_RATE = 16000

def pool_embeddings(embedded_states, pooling_method):
    """pool embeddings using the specified pooling method

    Args:
        embedded_states (tensor): tensor with shape [1, num_frames, embedding_size]
        pooling_method (string): e.g. mean pooling

    Raises:
        ValueError: unrecognised pooling method

    Returns:
        tensor: embedded states of shape [1, embedding_size]
    """
    if pooling_method == "mean":
        embedded_states = embedded_states.mean(axis=0)
    else:
        raise ValueError("UNRECOGNISED POOLING METHOD")
    return embedded_states

def pool_document_embeddings(embedded_states, window_size_ms, stride_ms, pooling_method):
    """generates new embeddings for a document by pooling embeddings in a sliding window. 
    The edges of the document are padded by half the window size. The window starts on the first
    frame of the document and moves by the stride. It ends when continuing to stride would 
    result in the window going past the last frame of the document (not including the padding).

    Args:
        embedded_states (tensor): tensor of shape [1, num_frames, embedding_size]
        window_size_ms (int): size of window in milliseconds
        stride_ms (int): size of stride in milliseconds
        pooling_method (string): e.g. mean pooling

    Raises:
        ValueError: window size must be odd

    Returns:
        tensor: tensor with the windowed and pooled embeddings
    """
    window_size, stride = get_window_size_and_stride_in_frames_from_ms(window_size_ms, stride_ms,
                                                                        SAMPLE_RATE)

    if window_size % 2 == 0:
        raise ValueError("Window size must be odd")
    length = embedded_states.shape[0]
    padded_embedding = F.pad(embedded_states, (0,0, window_size//2, window_size//2), 'constant', 0)

    output_states = torch.zeros((length-1)//stride + 1, embedded_states.shape[1])

    for i in range(window_size//2, length + window_size//2, stride):
        window = padded_embedding[i-window_size//2:i+window_size//2 + 1]
        pooled_window = pool_embeddings(window, pooling_method)
        output_states[i//stride - 1, :] = pooled_window

    return output_states

def pool_and_save_documents(raw_document_embedded_states_dir, new_document_embedded_states_dir, 
                            window_size_ms, stride_ms, pooling_method, n_parts=1, part=0):
    """For each document generate new embeddings using a sliding window and save the new embeddings.

    Args:
        raw_document_embedded_states_dir (string): directory containing raw document embeddings
        new_document_embedded_states_dir (string): directory to save the new document embeddings
        window_size_ms (int): size of window in milliseconds
        stride_ms (int): size of string in milliseconds
        pooling_method (string): e.g. mean pooling
    """
    files = sorted(os.listdir(raw_document_embedded_states_dir))
    file_part = split_list_into_n_parts_and_get_part(files, n_parts, part)
    for document_fname in file_part:
        print(document_fname)
        if "all_embeddings" in document_fname:
            continue
        raw_embedded_states = pkl.load(open(f"{raw_document_embedded_states_dir}/{document_fname}", "rb"))
        new_embedded_states = pool_document_embeddings(raw_embedded_states, window_size_ms, 
                                                       stride_ms, pooling_method)
        pkl.dump(new_embedded_states, open(f"{new_document_embedded_states_dir}/{document_fname}", "wb"))

def pool_and_save_queries(raw_query_embedded_states_dir, new_query_embedded_states_dir, pooling_method):
    """For each query, average pool the embeddings and save the new embedding.

    Args:
        raw_query_embedded_states_dir (string): location of directory containing raw query embeddings
        new_query_embedded_states_dir (string): location to save the new query embeddings
        pooling_method (string): e.g. mean pooling
    """
    files = sorted(os.listdir(raw_query_embedded_states_dir))
    for query_fname in files:
        raw_embedded_states = pkl.load(open(f"{raw_query_embedded_states_dir}/{query_fname}", "rb"))
        new_embedded_states = pool_embeddings(raw_embedded_states, pooling_method)
        pkl.dump(new_embedded_states, open(f"{new_query_embedded_states_dir}/{query_fname}", "wb"))
    

def get_window_size_and_stride_in_frames_from_ms(window_size_ms, stride_ms, sample_rate):
    """return the window size and stride in hubert frames given the 
        window size and stride in milliseconds. The hubert CNN compresses
        the input by a factor of 320, so one hubert frame is 320 samples.

    Args:
        window_size_ms (int): size of window in milliseconds
        stride_ms (int): size of stride in milliseconds
        sample_rate (int): sample rate of data

    Returns:
        int, int: window size and stride in hubert frames
    """
    window_size = int((window_size_ms * sample_rate) / (1000 * 320))
    stride = int((stride_ms * sample_rate) / (1000 * 320))

    if window_size % 2 == 0:
        window_size += 1

    return window_size, stride

def run(lay, window_size_ms, stride_ms, embedding_dir, 
        run_for_queries, run_for_documents, pooling_method="mean",
        multiple_query_vectors=False, n_parts=1, part=0):
    """run the pooling process for a given layer

    Args:
        lay (int): integer representing the layer
        window_size_ms (int): window size in milliseconds
        stride_ms (int): stride in milliseconds
        embedding_dir (string): embedding directory
        pooling_for (str, optional): "documents", "queries" or "documents and queries"
        determines which objects to pool. Defaults to "documents".
        pooling_method (str, optional): pooling method. Defaults to "mean".
    """
    document_prefix = f"{embedding_dir}/documents"
    query_prefix = f"{embedding_dir}/queries"

    print(f"Pooling embeddings for layer {lay}")

    raw_document_embedded_states_dir = f"{document_prefix}/{lay}/raw"
    raw_query_embedded_states_dir = f"{query_prefix}/{lay}/raw"

    new_document_embedded_states_dir = (f"{document_prefix}/{lay}/{pooling_method}_pooled_win_"
        f"{window_size_ms}ms_stride_{stride_ms}ms")
    if not multiple_query_vectors:
        new_query_embedded_states_dir = f"{query_prefix}/{lay}/{pooling_method}_pooled"
    else:
        new_query_embedded_states_dir = (f"{query_prefix}/{lay}/{pooling_method}_pooled_win_"
            f"{window_size_ms}ms_stride_{stride_ms}ms")

    make_dir(new_document_embedded_states_dir)
    make_dir(new_query_embedded_states_dir)
    
    if run_for_queries:
        if not multiple_query_vectors:
            pool_and_save_queries(raw_query_embedded_states_dir, new_query_embedded_states_dir,
            pooling_method)
        else:
            pool_and_save_documents(raw_query_embedded_states_dir, new_query_embedded_states_dir,
                                    window_size_ms, stride_ms, pooling_method)
        print(f"Pooled query embeddings for layer {lay}")
    if run_for_documents:
        pool_and_save_documents(raw_document_embedded_states_dir, new_document_embedded_states_dir,
                                window_size_ms, stride_ms, pooling_method, n_parts, part)
        print(f"Pooled document embeddings for layer {lay}")

if __name__ == "__main__":
    args = sys.argv
    if len(args)>1:
        window_size_ms = int(args[1])  # in milliseconds
        stride_ms = int(args[2])  # in milliseconds
        layer = int(args[3])
        run_for_queries = parse_boolean_input(args[4])
        run_for_documents = parse_boolean_input(args[5])
        n_parts = int(args[6])
        part = int(args[7])
    else:
        window_size_ms = 700  # in milliseconds
        stride_ms = 300  # in milliseconds
        layer = 9
        run_for_queries = True
        run_for_documents = True

    pooling_method = "mean"
    embedding_dir = "data/tamil/embeddings"
    # layer = int(sys.argv[1])
    # max_layers = 13
    multiple_query_vectors = True
    # select which types of files you want to run the script for

    run(layer, window_size_ms, stride_ms, embedding_dir, run_for_queries=run_for_queries,
        run_for_documents=run_for_documents, pooling_method=pooling_method, 
        multiple_query_vectors=multiple_query_vectors, n_parts=n_parts, part=part)
    # collate_embeddings(embedding_dir, layer, window_size_ms, stride_ms, 
    #     pooling_method=pooling_method)