import torch
from torch.nn.utils.rnn import pad_sequence
import time
import sys
import pickle as pkl
import mhubert_model.query_document_search as qds
from utils.common_functions_pytorch import print_memory_usage
from utils.common_functions import parse_boolean_input

def pad_normalise_from_list(tensor_list):
    """Take a list of tensors, pads them to the same length, then normalises them.
        Expected input is a list of 2d tensors of shape [rows, embedding_dimension].

    Args:
        list (list[tensor]): list of tensors

    Returns:
        tensor: tensor of normalised and padded tensors of shape [number_of_input_tensors,
        max_rows_of_input_tensors, embedding_dimension]
    """
    output_tensor = pad_sequence(tensor_list, batch_first=True)
    output_tensor_norms = output_tensor.norm(dim=2, p=2).unsqueeze(2)
    output_tensor = output_tensor/output_tensor_norms
    output_tensor = torch.nan_to_num(output_tensor, nan=0.0)

    return output_tensor

def pad_normalise_from_dict(dict):
    """Take a dictionary of filenames:tensors, combines the tensors and pads them to the same length, 
        then normalises them.

    Args:
        dict (dictionary(str:tensor)): dictionary of filenames:tensors

    Returns:
        tensor, list[string]: tensor of normalised and padded tensors of shape [number_of_input_tensors,
        max_rows_of_input_tensors, embedding_dimension], list of filenames
    """
    list_of_tensors = []
    names = list(dict.keys())
    for name in names:
        tensor = dict.pop(name)
        list_of_tensors.append(tensor)
    output_tensor = pad_normalise_from_list(list_of_tensors)

    return output_tensor, names

def read_size_order_file(size_order_file):
    names = []
    with open(size_order_file, "r") as f:
        for line in f:
            filename, _ = line.split(": ")
            filename = filename.strip()
            filename = filename.replace(".wav", ".pkl")
            names.append(filename)
    return names

def calculate_mem_usage_of_tensor_list_gb(tensor_list, padded_length):
    """calculate memory usage of a list of tensors in GB. Assumes tensors
        are padded to the same length and that they are 2d with
        the 0th axis the padded dimension. Assumes all tensors have the same
        size 1st axis.

    Args:
        tensor_list (list[tensor]): list of tensors
        padded_length (int): length the tensors will be padded to

    Returns:
        float: memory usage in gb
    """
    axis_1_size = tensor_list[0].shape[1]
    exp_memory = (padded_length * axis_1_size * 4 * len(tensor_list)) / (1024**3)
    return exp_memory

def pad_normalise_from_dict_with_size_order_batch(dictionary, size_order_file, max_mem_usage_per_batch_gb):
    """Takes a dictionary of filenames:tensors, combines the tensors and pads them to the same length,
        then normalises them. Also takes a file containing the size order of the tensors, and
        a max size per batch in gb. Tensors are padded and normalised in batches 
        according to the size order. 

    Args:
        dict (dictionary(str:tensor)): dictionary of filenames:tensors
        size_order_file (string): filename of the file containing the size order of the tensors,
        file must be formatted <filename>: <size> per line and should be in ascending order of size
        from top to bottom.
        max_mem_usage_per_batch_gb (float): max memory usage per batch in gb.

    Returns:
        list[tensor], list[list[string]]: list of tensor of normalised and padded tensors of shape 
        [number_of_input_tensors_in_batch, max_rows_of_input_tensors_in_batch, embedding_dimension]
        each element in tuple contains elements for one batch, list of lists of filenames, each list
        containing the filenames of the tensors in that batch.
    """

    names = read_size_order_file(size_order_file)
    output_list = []
    names_batched = []
    names_list = []
    tensor_batch_list = []
    padded_length = 0

    for i, name in enumerate(names):
        # name = name.replace("q_", "")
        if name not in dictionary.keys():
            print(f"WARNING: {name} not in dictionary, skipping.")
            if i < len(names) - 1:
                continue
        else:
            tensor = dictionary.pop(name)
            names_list.append(name)
            tensor_batch_list.append(tensor)
            padded_length = max(padded_length, tensor.shape[0])
            mem_usage_gb = calculate_mem_usage_of_tensor_list_gb(tensor_batch_list, padded_length)

        if (mem_usage_gb >= max_mem_usage_per_batch_gb) or (i == len(names) - 1):
            print(f"Creating batch of length: {len(names_list)}, size: {mem_usage_gb:.2f} GB")
            print_memory_usage()
            output_tensor = pad_normalise_from_list(tensor_batch_list)
            output_list.append(output_tensor)
            names_batched.append(names_list)
            tensor_batch_list = []
            names_list = []

    return output_list, names_batched

if __name__ == "__main__":
    args = sys.argv
    if len(args) > 1:
        window_size_ms = int(args[1])  # in milliseconds
        stride_ms = int(args[2])  # in milliseconds
        layer = int(args[3])
        run_for_queries = parse_boolean_input(args[4])
        run_for_documents = parse_boolean_input(args[5])
    else:
        window_size_ms = None  # in milliseconds
        stride_ms = None  # in milliseconds
        layer = 9
        run_for_queries = True
        run_for_documents = True

    folder = "tamil"
    embedding_dir = f"data/{folder}/embeddings"
    document_prefix = f"{embedding_dir}/documents"
    query_prefix = f"{embedding_dir}/queries"
    doc_size_order_file = f"data/{folder}/analysis/document_lengths.txt"
    # doc_size_order_file = f"data/{folder}/analysis/document_lengths_288.txt"
    query_size_order_file = f"data/{folder}/analysis/queries_lengths.txt"
    # query_size_order_file = f"data/{folder}/analysis/queries_lengths_nq_99_nd_288.txt"
    if window_size_ms is None or window_size_ms=="none":
        pooling_method = "none"
    else:
        pooling_method = "mean"

    query_multiple_vectors = True
    max_document_batch_size_gb = 0.1
    batch_query = True
    max_query_batch_size_gb = 0.1
    # set which types of files you want to run the script for

    print(f"Preprocessing files for layer {layer}")
    document_embedded_states_dir, query_embedded_states_dir, _ = \
        qds.get_embedding_and_results_dir(document_prefix, query_prefix, "", pooling_method, 
                                        layer, window_size_ms, stride_ms, query_multiple_vectors)

    # query_embedded_states_dir = f"{query_embedded_states_dir}_one_query"

    qds.check_if_dir_exists(document_embedded_states_dir)
    qds.check_if_dir_exists(query_embedded_states_dir)

    print(f"Document embedded states dir: {document_embedded_states_dir}")
    print(f"Query embedded states dir: {query_embedded_states_dir}")

    if run_for_queries:
        t1 = time.perf_counter()
        queries_embedded_states = qds.load_embeddings_from_dir(query_embedded_states_dir)
        t2 = time.perf_counter() 
        print(f"Time taken to load queries: {t2 - t1:.2f} seconds")

        t1 = time.perf_counter()
        if batch_query:
            batch_queries_padded_normalised, batch_names = pad_normalise_from_dict_with_size_order_batch(
                queries_embedded_states, query_size_order_file, max_query_batch_size_gb)
        else:
            queries_padded_normalised, names = pad_normalise_from_dict(queries_embedded_states)
        t2 = time.perf_counter() 
        print(f"Time taken to pad and normalise queries: {t2 - t1:.2f} seconds")

        t1 = time.perf_counter()
        if batch_query:
            pkl.dump((batch_queries_padded_normalised, batch_names), open(
                f"{query_embedded_states_dir}/all_embeddings_padded_normalised_batched.pkl", "wb"))
        else:
            pkl.dump((queries_padded_normalised, names), open(
                f"{query_embedded_states_dir}/all_embeddings_padded_normalised.pkl", "wb"))
        t2 = time.perf_counter() 
        print(f"Time taken to save queries: {t2 - t1:.2f} seconds")

    if run_for_documents:
        t1 = time.perf_counter()
        ranker = qds.Ranker(document_embedded_states_dir)
        document_embeddings = ranker.get_document_embeddings()
        t2 = time.perf_counter() 
        print(f"Time taken to load documents: {t2 - t1:.2f} seconds")
        print_memory_usage()

        t1 = time.perf_counter()
        batch_documents_padded_normalised, batch_names = pad_normalise_from_dict_with_size_order_batch(
            document_embeddings, doc_size_order_file, max_document_batch_size_gb)
        t2 = time.perf_counter() 
        print(f"Time taken to pad and normalise documents: {t2 - t1:.2f} seconds")

        t1 = time.perf_counter()
        pkl.dump((batch_documents_padded_normalised, batch_names), open(
            f"{document_embedded_states_dir}/all_embeddings_padded_normalised_batched.pkl", "wb"))
        t2 = time.perf_counter() 
        print(f"Time taken to save documents: {t2 - t1:.2f} seconds")

