"""Compute ranking of documents for all queries using vectorised operations."""
import time
import sys
import os
import pickle as pkl

import mhubert_model.query_document_search as qds
from mhubert_model.query_document_search_vectorised import compute_ranking 
from awe_model.ranking_preprocessing import batch_normalise_files
from utils.common_functions import parse_boolean_input

def clean_pkl_files(directory, exceptions):
    """Remove all .pkl files in a directory, except for those in the exceptions list.

    Args:
        directory (str): path of directory to clean.
        exceptions (list[str]): list of filenames to exclude from deletion.
    """
    for filename in os.listdir(directory):
        if filename.endswith(".pkl") and filename not in exceptions:
            os.remove(f"{directory}/{filename}")
    
    os.rmdir(directory)

def perform_search(document_prefix,
                    query_prefix,
                    results_dir_prefix, 
                    layer, 
                    document_query_suffix,
                    results_path,
                    load_query_embeddings_from_file,
                    load_doc_embeddings_from_file,
                    doc_size_order_file,
                    query_size_order_file,
                    max_document_batch_size_gb,
                    max_query_batch_size_gb,
                    device,
                    clean_files,
                    num_results_to_save=None):
    """Perform ranking of documents for all queries, assuming they have been embedded
    using the AWE model.

    Args:
        document_prefix (str): top level prefix for the directory containing document embeddings.
            Path is: {document_prefix}/{layer}/{document_query_suffix}.
        query_prefix (str): top level prefix for the directory containing query embeddings.
            Path is: {query_prefix}/{layer}/{document_query_suffix}.
        results_dir_prefix (str): top level prefix for the directory to save results.
            Path is: {results_dir_prefix}/{results_path}.
        layer (int): layer of mHuBERT model used to produce embeddings.
        document_query_suffix (str): suffix of directories containing document and query embeddings.
            Path is: {document_prefix}/{layer}/{document_query_suffix}. 
        results_path (str): path to results directory.
            Path is: {results_dir_prefix}/{results_path}.
        load_query_embeddings_from_file (bool): if true, system will try to load query embeddings from file,
            named "all_embeddings_padded_normalised_batched.pkl".
        load_doc_embeddings_from_file (bool): if true, system will try to load document embeddings from file,
            named "all_embeddings_padded_normalised_batched.pkl".
        doc_size_order_file (str): path to file containing list of documents in size order.
        query_size_order_file (str): path to file containing list of queries in size order.
        max_document_batch_size_gb (float): maximum size of document batch in GB.
        max_query_batch_size_gb (float): maximum size of query batch in GB.
        device (str): device to use for computations.
        clean_files (bool): set to true to clean any intermediate files.
        num_results_to_save (int, optional): limit on number of results to save, set to None to save all results.
            Defaults to None.
    """

    document_embedded_states_dir = \
        f"{document_prefix}/{layer}/{document_query_suffix}"
    query_embedded_states_dir = \
        f"{query_prefix}/{layer}/{document_query_suffix}"
    results_dir = \
        f"{results_dir_prefix}/{results_path}"

    qds.check_if_dir_exists(document_embedded_states_dir)
    qds.check_if_dir_exists(query_embedded_states_dir)
    qds.make_dir(results_dir)

    num_results = num_results_to_save if num_results_to_save is not None else "all"
    results_file = f"{results_dir}/results_{num_results}.txt"

    print(f"Document embedded states dir: {document_embedded_states_dir}")
    print(f"Query embedded states dir: {query_embedded_states_dir}")
    print(f"Results file: {results_file}")

    queries_fname = f"{query_embedded_states_dir}/all_embeddings_padded_normalised_batched.pkl"
    documents_fname = f"{document_embedded_states_dir}/all_embeddings_padded_normalised_batched.pkl"

    t1 = time.perf_counter()
    if load_query_embeddings_from_file:
        queries_embedded_states_batched, query_names_batched = pkl.load(open(queries_fname, "rb"))
    else:
        queries_embedded_states_batched, query_names_batched = batch_normalise_files(
            query_embedded_states_dir, query_size_order_file, max_query_batch_size_gb)
    if clean_files:
        clean_pkl_files(query_embedded_states_dir, ["all_embeddings_padded_normalised_batched.pkl"])
        print(f"Cleaned {query_embedded_states_dir}")
    t2 = time.perf_counter() 
    print(f"Time taken to load queries: {t2 - t1:.2f} seconds")

    t1 = time.perf_counter()
    if load_doc_embeddings_from_file:
        document_embedded_states_batched, document_names_batched = pkl.load(open(documents_fname, "rb"))
    else:
        document_embedded_states_batched, document_names_batched = batch_normalise_files(
            document_embedded_states_dir, doc_size_order_file, max_document_batch_size_gb)
    if clean_files:
        clean_pkl_files(document_embedded_states_dir, ["all_embeddings_padded_normalised_batched.pkl"])
        print(f"Cleaned {document_embedded_states_dir}")
    t2 = time.perf_counter() 
    print(f"Time taken to load documents: {t2 - t1:.2f} seconds")

    compute_ranking(queries_embedded_states_batched, query_names_batched, document_embedded_states_batched,
                    document_names_batched, results_file, device, num_results_to_save)

if __name__ == "__main__":
    args = sys.argv

    if len(args) > 1:
        language = args[1]
        layer = int(args[2])
        min_phone_seq_length = int(args[3])
        max_phone_seq_length = int(args[4])
        use_queries_cut_after_embedding = parse_boolean_input(args[5])
        document_query_suffix = args[6]
        results_path = args[7]
    else:
        language = "banjara"
        layer = 9
        min_phone_seq_length = 5
        max_phone_seq_length = 14
        use_queries_cut_after_embedding = False
        document_query_suffix = f"{min_phone_seq_length}_{max_phone_seq_length}"
        results_path = f"{layer}/tamil_train_3_9/{min_phone_seq_length}_{max_phone_seq_length}"

    scratch_prefix = f"/scratch/space1/tc062/sanjayb"
    # scratch_prefix = f"."
    embedding_dir = f"{scratch_prefix}/data/{language}/embeddings"
    document_prefix = f"{embedding_dir}/documents"
    if use_queries_cut_after_embedding:
        query_prefix = f"{embedding_dir}/queries_cut_after_embedding"
    else:
        query_prefix = f"{embedding_dir}/queries"
    results_dir_prefix = \
        f"data/{language}/results/awe/"

    load_doc_embeddings_from_file = False
    load_query_embeddings_from_file = False
    clean_files = True
    doc_size_order_file = f"data/{language}/analysis/document_lengths.txt"
    query_size_order_file = f"data/{language}/analysis/queries_lengths.txt"
    max_document_batch_size_gb = 0.05
    max_query_batch_size_gb = 0.1

    num_results_to_save = None  # set to None to save all results
    device = "cuda"

    print(f"Ranking documents for layer {layer}")

    perform_search(document_prefix,
                    query_prefix,
                    results_dir_prefix, 
                    layer, 
                    document_query_suffix,
                    results_path,
                    load_query_embeddings_from_file,
                    load_doc_embeddings_from_file,
                    doc_size_order_file,
                    query_size_order_file,
                    max_document_batch_size_gb,
                    max_query_batch_size_gb,
                    device,
                    clean_files,
                    num_results_to_save)