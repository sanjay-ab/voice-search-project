import os
import sys
import pickle as pkl
import torch
import torch.nn.functional as F
import multiprocessing as mp
from mhubert_model.collate_embeddings import collate_embeddings

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
                            window_size_ms, stride_ms, pooling_method):
    """For each document generate new embeddings using a sliding window and save the new embeddings.

    Args:
        raw_document_embedded_states_dir (string): directory containing raw document embeddings
        new_document_embedded_states_dir (string): directory to save the new document embeddings
        window_size_ms (int): size of window in milliseconds
        stride_ms (int): size of string in milliseconds
        pooling_method (string): e.g. mean pooling
    """
    files = sorted(os.listdir(raw_document_embedded_states_dir))
    for document_fname in files:
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

def make_dir(path):
    if not os.path.exists(f"{path}"):
        os.makedirs(f"{path}")

def run(lay, window_size_ms, stride_ms, embedding_dir, 
        run_for_queries, run_for_documents, pooling_method="mean",
        multiple_query_vectors=False):
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
                                window_size_ms, stride_ms, pooling_method)
        print(f"Pooled document embeddings for layer {lay}")

if __name__ == "__main__":
    pooling_method = "mean"
    window_size_ms = 700  # in milliseconds
    stride_ms = 300  # in milliseconds
    embedding_dir = "tamil_test_data/embeddings"
    # layer = int(sys.argv[1])
    max_layers = 13
    layer = 9
    multiple_query_vectors = True
    # select which types of files you want to run the script for
    run_for_queries = True
    run_for_documents = True

    for layer in range(max_layers):
        run(layer, window_size_ms, stride_ms, embedding_dir, run_for_queries=run_for_queries,
            run_for_documents=run_for_documents, pooling_method=pooling_method, 
            multiple_query_vectors=multiple_query_vectors)
        collate_embeddings(embedding_dir, layer, window_size_ms, stride_ms, 
            pooling_method=pooling_method)