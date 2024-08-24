import torch
import time
import pickle as pkl
import mhubert_model.query_document_search as qds
from utils.common_functions_pytorch import print_memory_usage

def compute_ranking(query_embeddings_batched, query_names_batched, document_embeddings_batched, 
                    document_names_batched, results_file, device, num_results_to_save=None):

    with torch.no_grad():

        num_doc_batches = len(document_names_batched)
        num_query_batches = len(query_embeddings_batched)
        print(f"Number of batches for queries: {num_query_batches}")
        print(f"Number of batches for documents: {num_doc_batches}")

        doc_names_combined = [] 
        query_names_combined = [] 
        similarity_combined = []
        print_memory_usage()
        t0 = time.perf_counter()
        for query_batch_num in range(num_query_batches):
            query_embeddings = query_embeddings_batched[query_batch_num]
            query_names_combined.extend(query_names_batched[query_batch_num])
            query_embeddings = query_embeddings.to(device)
            similarity_combined_query_batch = torch.empty(query_embeddings.shape[0], 0, 
                                                          dtype=torch.float32)

            print(f"Computing query batch {query_batch_num}")
            for doc_batch_num in range(num_doc_batches):
                document_embeddings = document_embeddings_batched[doc_batch_num]
                document_names = document_names_batched[doc_batch_num]
                if query_batch_num == num_query_batches - 1:
                    doc_names_combined.extend(document_names)

                document_embeddings = document_embeddings.to(device)

                t1 = time.perf_counter()
                product = torch.einsum("qve,dne->qdvn", query_embeddings, document_embeddings)
                t2 = time.perf_counter()
                print(f"Time taken to compute product for document batch {doc_batch_num}: {t2 - t1:.2f} s")
                print_memory_usage()

                maxes, _ = torch.max(product, dim=3)

                del document_embeddings
                del product

                num_non_zero = torch.count_nonzero(maxes, dim=2)
                similarity = torch.sum(maxes, dim=2) / num_non_zero
                similarity = similarity.to("cpu")
                similarity_combined_query_batch = torch.cat((similarity_combined_query_batch, 
                                                             similarity), dim=1)
            similarity_combined.append(similarity_combined_query_batch)
            del query_embeddings

        similarity_combined = torch.cat(similarity_combined, dim=0)

        if num_results_to_save is None or num_results_to_save > similarity_combined.shape[1]:
            num_results_to_save = similarity_combined.shape[1]

        top_values, top_indices = torch.topk(similarity_combined, num_results_to_save, 
                                             dim=1, sorted=True)
        t3 = time.perf_counter()
        print(f"Total time taken to compute result: {t3 - t0:.2f} s")

    with open(results_file, "w") as f:
        for i, query_name in enumerate(query_names_combined):
            f.write(f"Ranking for query: {query_name}\n")
            f.write(f"Document: Similarity\n")
            for j, value in enumerate(top_values[i]):
                f.write(f"{doc_names_combined[top_indices[i][j]]}: {value}\n")

def load_queries(fname, limit=None):
    queries_embedded_states, query_names = pkl.load(open(fname, "rb"))

    if limit is not None:
        queries_embedded_states = queries_embedded_states[:limit]
        query_names = query_names[:limit]

    return queries_embedded_states, query_names

if __name__ == "__main__":
    folder = "banjara"
    embedding_dir = f"data/{folder}/embeddings"
    document_prefix = f"{embedding_dir}/documents"
    query_prefix = f"{embedding_dir}/queries"
    results_dir_prefix = f"data/{folder}/results/raw_hubert"
    layer = 9
    pooling_method = "none"
    window_size_ms = None  # in milliseconds
    stride_ms = None  # in milliseconds
    query_multiple_vectors = True
    query_batched = True
    query_limit = None
    num_results_to_save = None  # set to None to save all results
    device = "cuda"

    n_parts = 1
    # part = sys.argv[1]
    part = 0

    if n_parts == 1:
        part = "all"

    print(f"Ranking documents for layer {layer}")
    print(f"Query limit: {query_limit}")
    document_embedded_states_dir, query_embedded_states_dir, results_dir = \
        qds.get_embedding_and_results_dir(document_prefix, query_prefix, results_dir_prefix, pooling_method, 
                                        layer, window_size_ms, stride_ms, query_multiple_vectors)

    qds.check_if_dir_exists(document_embedded_states_dir)
    qds.check_if_dir_exists(query_embedded_states_dir)
    qds.make_dir(results_dir)

    if query_limit is not None:
        results_file = f"{results_dir}/results_{part}_limit_{query_limit}.txt"
    else:
        results_file = f"{results_dir}/results_{part}.txt"

    print(f"Document embedded states dir: {document_embedded_states_dir}")
    print(f"Query embedded states dir: {query_embedded_states_dir}")
    print(f"Results file: {results_file}")

    if query_batched:
        queries_fname = f"{query_embedded_states_dir}/all_embeddings_padded_normalised_batched.pkl"
    else:
        queries_fname = f"{query_embedded_states_dir}/all_embeddings_padded_normalised.pkl"
    documents_fname = f"{document_embedded_states_dir}/all_embeddings_padded_normalised_batched.pkl"

    t1 = time.perf_counter()
    if query_batched:
        queries_embedded_states_batched, query_names_batched = pkl.load(open(queries_fname, "rb"))
        if query_limit is not None:
            print(f"Note: query limit is not implemented with batched queries.")
    else:
        queries_embedded_states, query_names = load_queries(queries_fname, query_limit)
        queries_embedded_states_batched = [queries_embedded_states]
        query_names_batched = [query_names]

    t2 = time.perf_counter() 
    print(f"Time taken to load queries: {t2 - t1:.2f} seconds")

    t1 = time.perf_counter()
    document_embedded_states_batched, document_names_batched = pkl.load(open(documents_fname, "rb"))
    t2 = time.perf_counter() 
    print(f"Time taken to load documents: {t2 - t1:.2f} seconds")

    compute_ranking(queries_embedded_states_batched, query_names_batched, document_embedded_states_batched,
                    document_names_batched, results_file, device, num_results_to_save)