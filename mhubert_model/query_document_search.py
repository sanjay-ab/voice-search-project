"""Perform query-document search without vectorised operations."""
import os
import sys
import time
import pickle as pkl

import torch

from utils.common_functions import make_dir, split_list_into_n_parts_and_get_part, check_if_dir_exists
from utils.common_functions_pytorch import print_memory_usage

class Ranker:
    """Used to rank documents based on similarity to a query. Doesn't use vectorised operations."""

    def __init__(self, document_embedded_states_dir, similarity_metric="cosine", 
                 results_file_name=None, method="w", force_dont_use_all_embeddings_file=False):
        """Initialise ranker with document embeddings and similarity metric

        Args:
            document_embedded_states_dir (str): directory of document embeddings
            similarity_metric (str, optional): similarity metric to use. Defaults to "cosine".
            results_file_name (str, optional): path of file to save results. Defaults to None.
            method (str, optional): method of how to open results file e.g w or a+. Defaults to "w".
            force_dont_use_all_embeddings_file (bool, optional): force ranker to not use file containing 
                all embeddings.

        Raises:
            ValueError: for an unrecognised similarity metric
        """
        self.force_dont_use_all_embeddings_file = force_dont_use_all_embeddings_file
        self.document_embedded_states_dir = document_embedded_states_dir
        self.all_document_embeddings_fname = f"all_embeddings.pkl"  # name of file containing all document embeddings

        self.load_document_embeddings()

        if similarity_metric == "cosine":
            self.similarity_metric = torch.nn.CosineSimilarity(dim=1)
        else:
            raise ValueError("Unrecognised similarity metric") 

        self.results_file = None
        if results_file_name is not None:
            self.open_results_file(results_file_name, method)
    
    def load_document_embeddings(self):
        """Load all document embeddings into self.document_embeddings from a directory."""

        self.document_files = sorted(os.listdir(self.document_embedded_states_dir))
        all_embeddings_file_exists = False
        if self.all_document_embeddings_fname in self.document_files:
            all_embeddings_file_exists = True

        self.document_embeddings = {}
        if all_embeddings_file_exists and (not self.force_dont_use_all_embeddings_file):
            print(f"Loading all document embeddings from {self.all_document_embeddings_fname}")
            self.document_embeddings = pkl.load(open(
                f"{self.document_embedded_states_dir}/{self.all_document_embeddings_fname}", "rb"))
        else:
            print(f"Loading all document embeddings individually")
            for file in self.document_files:
                if "all_embeddings" in file: continue
                self.document_embeddings[file] = pkl.load(open(
                    f"{self.document_embedded_states_dir}/{file}", "rb"))
    
    def get_document_embeddings(self):
        return self.document_embeddings
    
    def save_document_embedding(self):
        pkl.dump(self.document_embeddings, 
            open(f"{self.document_embedded_states_dir}/{self.all_document_embeddings_fname}", "wb"))

    def open_results_file(self, results_file_name, method="w"):
        """Open results file using the specified method. If a results file is already open, close it
        before opening the new one.

        Args:
            results_file_name (str): path of file to save results
            method (str, optional): open method for results file. Defaults to "w".
        """
        if self.results_file is not None:
            print((f"Warning results file {self.results_file} already open, "
                  f"closing it before opening {results_file_name}"))
            self.results_file.close()

        self.results_file = open(results_file_name, method)
    
    def close_results_file(self):
        self.results_file.close()

    def calculate_similarity(self, query_embedding, single_document_embeddings):
        """Calculates similarities (defafult cosine sim) between a query and all documents. 

        Args:
            query_embedding (tensor): single vector embedding for one query, does not support 
                multiple vectors for each query. shape [1, embedding_size]
            single_document_embeddings (tensor): vector embeddings for a single document,
                shape [1, time, embedding_size]

        Returns:
            tensor: similarities between vectors at each time step of the document and the single query vector
        """
        query_embedding = query_embedding.unsqueeze(0)
        similarities = self.similarity_metric(query_embedding, single_document_embeddings)
        return similarities

    def calculate_similarity_for_graphing(self, query_embedding, single_document_fname=None):
        """Returns similarities between a query and a document. Uses embeddings of specific document
        if a filename is passed in, otherwise uses the document embeddings of the document
        that matches best with the query.

        Args:
            query_embedding (tensor): tensor of the single vector query embedding, doesn't support 
                multiple vectors per query - shape [1, embedding_size]
            single_document_fname (str, optional): filename of a document to match against. Defaults to None.

        Returns:
            tuple(tensor, str): tensor is the similarities of query against specified document or against best matching document,
                str is the name of the best matching document - None if single_document_fname is passed in.
        """
        query_embedding = query_embedding.unsqueeze(1)

        if single_document_fname is not None:
            similarities = self.similarity_metric(query_embedding, 
                                                  self.document_embeddings[single_document_fname])
            return similarities[0].detach().numpy(), None

        best_similarity = 0
        best_document = ""
        best_similarities = []
        for document, embeddings in self.document_embeddings.items():
            similarities = self.calculate_similarity(query_embedding, embeddings)
            similarity = similarities.max().item()
            if similarity > best_similarity:
                best_similarity = similarity
                best_document = document
                best_similarities = similarities
        return best_similarities[0].detach().numpy(), best_document



    def rank_documents_for_query(self, query_embedding):
        """Ranks documents based on similarity to query. 

        Args:
            query_embedding (tensor): single vector query embedding, doesn't support
                multi-vector query embeddings, shape [1, embedding_size]

        Returns:
            dict{str \: float}: dictionary of document names and their similarity to the query. 
        """
        document_similarities = {}
        for document in self.document_files:
            embeddings = self.document_embeddings[document]
            similarities = self.calculate_similarity(query_embedding, embeddings)
            similarity = similarities.max().item()
            document_similarities[document] = similarity
        ranked_documents = dict(sorted(document_similarities.items(), 
                                       key=lambda x: x[1], reverse=True)) 
        
        return ranked_documents
    
    def rank_documents_for_query_multiple_vectors(self, query_embedding):
        """Ranks documents based on similarity to query. Query is a sequence of vectors.

        Args:
            query_embedding (tensor): shape [time, embedding_size]

        Returns:
            dict{str \: float}: dictionary of document names and their similarity to the query. 
        """
        document_similarities = {}
        for document in self.document_files:
            if "all_embeddings" in document: continue
            d_embeddings = self.document_embeddings[document]
            similarity_sum = 0
            for q_embedding in query_embedding:
                similarities = self.calculate_similarity(q_embedding, d_embeddings)
                similarity = similarities.max().item()
                similarity_sum += similarity
            document_similarities[document] = similarity_sum / query_embedding.shape[0]
        ranked_documents = dict(sorted(document_similarities.items(), 
                                       key=lambda x: x[1], reverse=True)) 
        
        return ranked_documents

    def save_ranking(self, query_name, ranking, num_to_save=10):
        """Save ranking for one query.

        Args:
            query_name (str): name of the query to save ranking for
            ranking (dict{str:float}): dictionary of document names and their similarity to the query
            num_to_save (int, optional): number of results to save for the query. Defaults to 10.

        Raises:
            ValueError: if results file is not open
        """
        if self.results_file is None:
            raise ValueError("Results file not open")

        self.results_file.write(f"Ranking for query: {query_name}\n")
        self.results_file.write(f"Document: Similarity\n")
        num = 0
        for document, similarity in ranking.items():
            self.results_file.write(f"{document}: {similarity}\n")
            num += 1
            if num >= num_to_save: break
    
    def rank_and_save_documents_for_query(self, query_name, query_embedding):
        if len(query_embedding.shape) == 1:
            ranking = self.rank_documents_for_query(query_embedding)
        else:
            ranking = self.rank_documents_for_query_multiple_vectors(query_embedding)
        self.save_ranking(query_name, ranking)

def load_embeddings_from_dir(directory, limit=None, all_embeddings_fname="all_embeddings.pkl"):
    """Load embeddings from a directory.

    Args:
        directory (str): directory to load embeddings from
        limit (int, optional): number of embeddings to load. Defaults to None.
        all_embeddings_fname (str, optional): name of file that contains all the embeddings in the 
            directory collated together. Defaults to "all_embeddings.pkl".

    Returns:
        dict{str \: tensor}: dictionary of file names and their embeddings
    """
    files = sorted(os.listdir(directory))
    if all_embeddings_fname in files:
        embeddings = pkl.load(open(f"{directory}/{all_embeddings_fname}", "rb"))
        return embeddings

    embeddings = {}
    for i, file in enumerate(files):
        if limit is not None:
            if i >= limit: break
        if "all_embeddings" in file: continue
        embeddings[file] = pkl.load(open(f"{directory}/{file}", "rb"))
    return embeddings

def get_embedding_and_results_dir(document_prefix, query_prefix, results_dir_prefix, 
                                  pooling_method, layer, window_size_ms, stride_ms, 
                                  query_multiple_vectors):
    """Helper method to get the directories for the embeddings and results using the 
        specified parameters. Particularly useful when a window is used."""
    if pooling_method == "none" or pooling_method is None:
        document_embedded_states_dir = (f"{document_prefix}/{layer}/raw")
    else:
        document_embedded_states_dir = (f"{document_prefix}/{layer}/{pooling_method}_pooled_win_"
            f"{window_size_ms}ms_stride_{stride_ms}ms")
    if not query_multiple_vectors:
        query_embedded_states_dir = f"{query_prefix}/{layer}/{pooling_method}_pooled/"
    else:
        if pooling_method == "none" or pooling_method is None:
            query_embedded_states_dir = (f"{query_prefix}/{layer}/raw")
        else:
            query_embedded_states_dir = (f"{query_prefix}/{layer}/{pooling_method}_pooled_win_"
                f"{window_size_ms}ms_stride_{stride_ms}ms")

    if not query_multiple_vectors:
        results_dir = (f"{results_dir_prefix}/{layer}/{pooling_method}_pooled_win_{window_size_ms}ms_"
                        f"stride_{stride_ms}ms")
    else:
        if pooling_method == "none" or pooling_method is None:
            results_dir = (f"{results_dir_prefix}/{layer}/raw_multiple_q_vecs")
        else:
            results_dir = (f"{results_dir_prefix}/{layer}/{pooling_method}_pooled_win_{window_size_ms}ms_"
                            f"stride_{stride_ms}ms_multiple_q_vecs")
    return document_embedded_states_dir, query_embedded_states_dir, results_dir

if __name__ == "__main__":
    general_prefix = "digit_recog_tests/embeddings"
    document_prefix = f"{general_prefix}/documents"
    query_prefix = f"{general_prefix}/queries"
    results_dir_prefix = "digit_recog_tests/results"
    layers = ["9"]
    pooling_method = "none"
    window_size_ms = 500
    stride_ms = 200
    query_multiple_vectors = True  # set to True if each query embedding is composed of multiple vectors
    query_limit = None  # set to limit the number of queries used, None = no limit

    n_parts = 1  # split the queries into n parts and only process one part
    # useful for parallelising the ranking of queries
    if n_parts > 1:
        part = sys.argv[1]
    else:
        part = "all"

    for layer in layers:
        print(f"Ranking documents for layer {layer}")
        document_embedded_states_dir, query_embedded_states_dir, results_dir = \
            get_embedding_and_results_dir(document_prefix, query_prefix, results_dir_prefix, 
                                          pooling_method, layer, window_size_ms, stride_ms, 
                                          query_multiple_vectors)
        
        if query_limit is not None:
            results_file = f"{results_dir}/results_{part}_limit_{query_limit}.txt"
        else:
            results_file = f"{results_dir}/results_{part}.txt"

        check_if_dir_exists(document_embedded_states_dir)
        check_if_dir_exists(query_embedded_states_dir)
        make_dir(results_dir)

        print(f"Document embedded states dir: {document_embedded_states_dir}")
        print(f"Query embedded states dir: {query_embedded_states_dir}")
        print(f"Results file: {results_file}")

        t1 = time.perf_counter()
        queries_embedded_states = load_embeddings_from_dir(query_embedded_states_dir, query_limit)
        t2 = time.perf_counter() 
        print(f"Time taken to load queries: {t2 - t1:.2f} seconds")

        t1 = time.perf_counter()
        ranker = Ranker(document_embedded_states_dir, results_file_name=results_file, method="w")
        t2 = time.perf_counter() 
        print(f"Time taken to load documents: {t2 - t1:.2f} seconds")

        print_memory_usage()

        query_names = list(queries_embedded_states.keys())
        query_list_part = split_list_into_n_parts_and_get_part(query_names, n_parts, part)

        t1 = time.perf_counter()
        for query_name in query_list_part:
            print(f"Ranking documents for query {query_name}")
            embedding = queries_embedded_states[query_name]
            ranker.rank_and_save_documents_for_query(query_name, embedding)
            t = time.perf_counter()
            print(f"Finished ranking documents for query {query_name}")
            print(f"Time taken: {t - t1:.2f} seconds")
        t2 = time.perf_counter() 
        print(f"Time taken to rank documents: {t2 - t1:.2f} seconds")
        ranker.close_results_file()
            # print(query)
            # similarities, _ = ranker.calculate_similarity_for_graphing(embedding, "s2556413_test.wav")
            # plt.plot(similarities)
            # plt.show()
