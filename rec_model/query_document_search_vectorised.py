import sys
import os
from awe_model.query_document_search_vectorised import perform_search

def clean_pkl_files(directory, exceptions):
    for filename in os.listdir(directory):
        if filename.endswith(".pkl") and filename not in exceptions:
            os.remove(f"{directory}/{filename}")
    
    os.rmdir(directory)

if __name__ == "__main__":
    args = sys.argv

    if len(args) > 1:
        language = args[1]
        layer = int(args[2])
        document_query_suffix = args[3]
        results_path = args[4]
    else:
        language = "banjara"
        layer = 9
        document_query_suffix = f"5_14_test_rec_one_query"
        results_path = f"train_tamil/{layer}/test"

    scratch_prefix = f"/scratch/space1/tc062/sanjayb"
    clean_files = True
    embedding_dir = f"{scratch_prefix}/data/{language}/embeddings"
    document_prefix = f"{embedding_dir}/documents"
    query_prefix = f"{embedding_dir}/queries"
    results_dir_prefix = \
        f"data/{language}/results/rec/"

    load_query_embeddings_from_file = False
    load_doc_embeddings_from_file = False
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