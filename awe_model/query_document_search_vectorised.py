import time
import sys
import os
import pickle as pkl
import mhubert_model.query_document_search as qds
from mhubert_model.query_document_search_vectorised import compute_ranking 
from awe_model.ranking_preprocessing import batch_normalise_files

def clean_pkl_files(directory, exceptions):
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
                    num_results_to_save,
                    device,
                    clean_files):

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
        document_query_suffix = args[5]
        results_path = args[6]
    else:
        language = "banjara"
        layer = 9
        min_phone_seq_length = 5
        max_phone_seq_length = 14
        document_query_suffix = f"{min_phone_seq_length}_{max_phone_seq_length}"
        results_path = f"{layer}/tamil_train_3_9/{min_phone_seq_length}_{max_phone_seq_length}"

    scratch_prefix = f"/scratch/space1/tc062/sanjayb"
    embedding_dir = f"{scratch_prefix}/data/{language}/embeddings"
    document_prefix = f"{embedding_dir}/documents"
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
                    num_results_to_save,
                    device,
                    clean_files)