import time
import pickle as pkl
from torch.nn.utils.rnn import pad_sequence
import mhubert_model.query_document_search as qds
from mhubert_model.ranking_preprocessing import pad_normalise_from_dict_with_size_order_batch
from utils.common_functions_pytorch import print_memory_usage

def batch_normalise_files(embedded_states_dir, size_order_file, max_batch_size_gb):
    t1 = time.perf_counter()
    embedded_states = qds.load_embeddings_from_dir(embedded_states_dir)
    t2 = time.perf_counter() 
    print(f"Time taken to load files: {t2 - t1:.2f} seconds")

    t1 = time.perf_counter()
    batch_padded_normalised, batch_names = pad_normalise_from_dict_with_size_order_batch(
        embedded_states, size_order_file, max_batch_size_gb)
    t2 = time.perf_counter() 
    print(f"Time taken to pad and normalise files: {t2 - t1:.2f} seconds")
    
    return batch_padded_normalised, batch_names

if __name__ == "__main__":
    language = "tamil"
    embedding_dir = f"data/{language}/embeddings"
    document_prefix = f"{embedding_dir}/documents"
    query_prefix = f"{embedding_dir}/queries"
    doc_size_order_file = f"data/{language}/analysis/document_lengths.txt"
    query_size_order_file = f"data/{language}/analysis/queries_lengths.txt"
    layer = 9
    max_document_batch_size_gb = 0.01
    max_query_batch_size_gb = 0.1
    min_phone_seq_length = 3
    max_phone_seq_length = 9
    # set which types of files you want to run the script for
    run_for_queries = True
    run_for_documents = True
    document_query_suffix = f"half_lower_lr_window_{min_phone_seq_length}_{max_phone_seq_length}"

    print(f"Preprocessing files for layer {layer}")
    document_embedded_states_dir = \
        f"{document_prefix}/{layer}/{document_query_suffix}"
    query_embedded_states_dir = \
        f"{query_prefix}/{layer}/{document_query_suffix}"

    qds.check_if_dir_exists(document_embedded_states_dir)
    qds.check_if_dir_exists(query_embedded_states_dir)

    print(f"Document embedded states dir: {document_embedded_states_dir}")
    print(f"Query embedded states dir: {query_embedded_states_dir}")

    if run_for_queries:
        print(f"QUERIES:")
        batch_queries_padded_normalised, batch_names = batch_normalise_files(
            query_embedded_states_dir, query_size_order_file, max_query_batch_size_gb)

        t1 = time.perf_counter()
        pkl.dump((batch_queries_padded_normalised, batch_names), open(
            f"{query_embedded_states_dir}/all_embeddings_padded_normalised_batched.pkl", "wb"))
        t2 = time.perf_counter() 
        print(f"Time taken to save queries: {t2 - t1:.2f} seconds")

    if run_for_documents:
        print(f"DOCUMENTS:")
        batch_documents_padded_normalised, batch_names = batch_normalise_files(
            document_embedded_states_dir, doc_size_order_file, max_document_batch_size_gb)

        t1 = time.perf_counter()
        pkl.dump((batch_documents_padded_normalised, batch_names), open(
            f"{document_embedded_states_dir}/all_embeddings_padded_normalised_batched.pkl", "wb"))
        t2 = time.perf_counter() 
        print(f"Time taken to save documents: {t2 - t1:.2f} seconds")

