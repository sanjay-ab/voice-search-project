import os
import re
import sys
import csv
from collections import defaultdict

import mhubert_model.query_document_search as qds
# from mhubert_model.query_document_search import get_embedding_and_results_dir
from utils.split_banjara_queries_documents import banjara_get_data_dicts
from utils.graph_banjara_data import banjara_graph_count_order_bar_chart
from utils.examine_datasets import clean_string
from utils.common_functions import get_wav_file_length

def get_query_results_dict(results_file, limit=None):
    """Get query results from a results file. The results file should be in the format:
    1st line: "Ranking for query: [query_basename]", 
    2nd line: "Document: Similarity", 
    following lines: "[document_basename]: [similarity]", until all documents are exhausted,
    then this repeats for the next query, until all queries are exhausted.

    Args:
        results_file (str): path to the results file
        limit (int, optional): limit the number of results per query to consider
            Set to None to consider all results. Defaults to None.

    Returns:
        dict{str \: tuple(list[str], list[float])}: a dictionary with query basenames as keys, and
            as values: a tuple with 2 lists: an ordered list of document basenames and an ordered list
            of corresponding document similarities to the query. Both lists ordered by similarity.     
    """
    query_results_dict = {}
    skip_next_line = False
    counter = 0
    with open(results_file, "r") as f:
        for line in f:
            if skip_next_line:
                skip_next_line = False
                continue
            if "query" in line:
                if limit is not None:
                    if counter >= limit:
                        break
                    counter += 1
                _, query = line.split(": ")
                query = clean_string(query)
                query_results_dict[query] = ([], [])
                skip_next_line = True
                continue
            doc, similarity = line.split(": ")
            doc = clean_string(doc)
            similarity = float(similarity)
            query_results_dict[query][0].append(doc)
            query_results_dict[query][1].append(similarity)
    
    return query_results_dict

def count_if_query_matches_itself(query_results_dict):
    """Count how many times a query matches with itself (for Banjara) or
    with the document it was extracted from (for Tamil) in the top 1 or 10 results.

    Args:
        query_results_dict (dict{str \: tuple(list[str], list[float])}): a dictionary
            with query basenames as keys, and as values: a tuple with 2 lists: an
            ordered list of document basenames and an ordered list of corresponding
            document similarities to the query. Both lists ordered by similarity.

    Returns:
        tuple(int, int): number of queries whose best match is itself, number of queries
            who match with itself in the top 10 matches.
    """
    num_queries_best_match_is_itself = 0
    num_queries_that_match_with_itself_in_top_10 = 0

    for query, results in query_results_dict.items():
        query = re.sub(r"_\d+$", "", query)
        document_results = results[0]
        if query in document_results[:10]:
            num_queries_that_match_with_itself_in_top_10 += 1
        if query == document_results[0]:
            num_queries_best_match_is_itself += 1
    
    return num_queries_best_match_is_itself, num_queries_that_match_with_itself_in_top_10

def extract_gold_labels_for_queries(reference_file, language):
    """Extract the correct documents for each query from a reference file.

    Args:
        reference_file (str): path to reference file, each line is assumed to be 
            formatted as: "[query] [tab] [correct_document_1] [tab] [correct_document_2] ..." for banjara, 
            and: "[query] [tab] [label] [tab] [correct_document_1] [tab] [correct_document_2] ..." for tamil.
        language (str): language of the reference file. Either "tamil" or "banjara"

    Returns:
        dict{str \: list[str]}: dictionary with query basenames as keys and list of document basenames
            as values. The documents are those that are correct matches for the query.
    """
    gold_documents_for_queries_dict = {}
    if language == "tamil":
        start_idx = 2
    elif language == "banjara":
        start_idx = 1
    with open(reference_file, "r") as f:
        for line in f:
            split_line = line.split("\t")
            query = split_line[0]
            query = clean_string(query)
            documents = split_line[start_idx:-1]
            gold_documents_for_queries_dict[query] = documents
    
    return gold_documents_for_queries_dict

def calculate_correct_results(query_result_dict, gold_documents_for_queries_dict):
    """Given a dictionary with query results and a dictionary with correct documents for queries,
    calculate how many queries have at least one correct result in the top 5 and top 10 results.

    Args:
        query_result_dict (dict{str \: tuple(list[str], list[float])}): a dictionary
            with query basenames as keys, and as values: a tuple with 2 lists: an
            ordered list of document basenames and an ordered list of corresponding
            document similarities to the query. Both lists ordered by similarity.
        gold_documents_for_queries_dict (dict{str \: list[str]}): dictionary with 
            query basenames as keys and list of document basenames as values. The documents
            are those that are correct matches for the query.

    Returns:
        int, int: number of queries with at least one correct result in the top 5 and top 10 results.
    """
    num_queries_with_result_in_top_10 = 0
    num_queries_with_result_in_top_5 = 0

    for query, results in query_result_dict.items():
        document_results = results[0].copy()
        if query in document_results:
            document_results.remove(query)
        for i, result in enumerate(document_results):
            if result in gold_documents_for_queries_dict[query]:
                if  i<5:
                    num_queries_with_result_in_top_5 += 1
                if i<10:
                    num_queries_with_result_in_top_10 += 1
                break
    return num_queries_with_result_in_top_5, num_queries_with_result_in_top_10

def get_label(file_basename, language, banjara_label_directory=None, tamil_label_dict=None):
    """Get label for a file basename.

    Args:
        file_basename (str): file basename
        language (str): language of the file basename. Either "banjara" or "tamil"
        banjara_label_directory (str, optional): path to directory with label files. Defaults to None.
        tamil_label_dict (dict{str:str}, optional): dictionary mapping file basenames to
            labels for tamil. Defaults to None.

    Returns:
        str: label for the file basename
    """
    if language == "banjara":
        label_fname = f"{banjara_label_directory}/{file_basename}.label"
        with open(label_fname, "r") as f:
            label = f.readline().strip()
    elif language == "tamil":
        label = tamil_label_dict[file_basename]
    return label

def read_tamil_label_file(reference_file):
    """Read reference file and extract labels for tamil documents.

    Args:
        reference_file (str): path to reference file with labels for tamil queries and documents

    Returns:
        dict{str \: str}: defaultdict with recording basenames as keys and labels as values
    """
    tamil_label_dict = defaultdict(lambda: "label unknown")
    with open(reference_file, "r") as f:
        for line in f:
            split_line = line.split("\t")
            query = split_line[0]
            query = clean_string(query)
            label = split_line[1]
            documents = split_line[2:-1]
            tamil_label_dict[query] = label
            for document in documents:
                tamil_label_dict[document] = label
    return tamil_label_dict

def write_analysis_file(analysis_file, query_results_dict, gold_documents_for_queries_dict, 
                        query_audio_dir, document_audio_dir, language, avg_precision_dict_at_5, 
                        avg_precision_dict_all, banjara_label_dir=None, tamil_reference_file=None):
    """Write an analysis file with information about the query results.

    Args:
        analysis_file (str): path to the analysis file
        query_results_dict (dict{str \: tuple(list[str], list[float])}): a dictionary with query 
            basenames as keys, and as values: a tuple with 2 lists: an ordered list of document
            basenames and an ordered list of corresponding document similarities to the query.
            Both lists ordered by similarity.
        gold_documents_for_queries_dict (dict{str \: list[str]}): dictionary with query basenames
            as keys and list of document basenames as values. The documents are those that are
            correct matches for the query.
        query_audio_dir (str): directory with query audio files
        document_audio_dir (str): directory with document audio files
        language (str): language of the data. Either "banjara" or "tamil"
        avg_precision_dict_at_5 (dict{str:float}): dictionary with query basenames as keys and
            average precision at 5 as values.
        avg_precision_dict_all (dict{str:float}): dictionary with query basenames as keys and
            average precision overall as values.
        banjara_label_dir (str, optional): path to directory containing banjara labels. Defaults to None.
        tamil_reference_file (str, optional): path to tamil reference file. Defaults to None.
    """
    if language not in ["banjara", "tamil"]:
        raise ValueError("Language must be either 'banjara' or 'tamil'.")

    if banjara_label_dir is None and language == "banjara":
        raise ValueError("Banjara label directory must be provided.")
    if tamil_reference_file is None and language == "tamil":
        raise ValueError("Tamil reference file must be provided.")

    if language == "tamil":
        tamil_label_dict = read_tamil_label_file(tamil_reference_file)
    else:
        tamil_label_dict = None
    
    with open(analysis_file, "w") as f:
        for query, results in query_results_dict.items():
            document_results = results[0]
            document_similarities = results[1]

            if language == "tamil":
                query_fname = f"q_{query}.wav"
            else:
                query_fname = f"{query}.wav"

            query_duration = get_wav_file_length(f"{query_audio_dir}/{query_fname}")

            query_label = get_label(query, language, banjara_label_dir, tamil_label_dict)

            num_document_avail_for_match = len(gold_documents_for_queries_dict[query])

            num_correct_matches_in_top_5 = 0
            counter = 0
            for document in document_results:
                if document == query: continue
                if counter >= 5: break
                if document in gold_documents_for_queries_dict[query]:
                    num_correct_matches_in_top_5 += 1
                counter += 1
            
            avg_precision_at_5 = avg_precision_dict_at_5[query]
            avg_precision_all = avg_precision_dict_all[query]

            f.write((f"QUERY: {query}, Number of matches in top 5: {num_correct_matches_in_top_5}, "
                     f"Average Precision at 5: {avg_precision_at_5}, "
                     f"Average Precision Overall: {avg_precision_all}, "
                     f"Label: {query_label}, Duration: {query_duration} s, "
                     f"Num of possible matching docs: {num_document_avail_for_match}\n"))

            f.write("MATCHED DOCUMENTS:\n")
            for i, document in enumerate(document_results):
                document_duration = get_wav_file_length(f"{document_audio_dir}/{document}.wav")
                document_label = get_label(document, language, banjara_label_dir, tamil_label_dict)
                document_similarity = document_similarities[i]
                if document == query:
                    correct_match = "Same as query"
                else:
                    correct_match = document in gold_documents_for_queries_dict[query]
                f.write((f"Document: {document}, Label: {document_label}, Duration: {document_duration} s, "
                         f"Similarity: {document_similarity}, Correct match: {correct_match}\n"))

            f.write("ALL GOLD DOCUMENTS::\n")
            for document in gold_documents_for_queries_dict[query]:
                document_duration = get_wav_file_length(f"{document_audio_dir}/{document}.wav")
                document_label = get_label(document, language, banjara_label_dir, tamil_label_dict)
                if document in document_results:
                    i = document_results.index(document)
                    document_similarity = document_similarities[i]
                else:
                    document_similarity = "unknown"
                f.write((f"Document: {document}, Label: {document_label}, Duration: {document_duration} s, "
                         f"Similarity: {document_similarity}\n"))
            f.write("\n")

def calculate_mean_average_precision(query_results_dict, gold_documents_for_queries_dict, 
                                     num_results_to_consider=None):
    """Calculate the mean average precision for the given query results.

    Args:
        query_results_dict (dict{str \: tuple(list[str], list[float])}): dictionary with query basenames
            as keys and tuples with 2 lists: a list of document basenames and a list of corresponding document 
            similarities, as values. The documents are those that are returned by the search for the query.
        gold_documents_for_queries_dict (dict{str:list[str]}): dictionary with query basenames as keys
            and list of document basenames as values. The documents are those that are correct matches 
            for the query.
        num_results_to_consider (int, optional): specifies the number of results to consider for the MAP
            calculation. If set to None then all results are included. Defaults to None.

    Returns:
        tuple(float, dict{str \: float}): the MAP score, dictionary mapping query basenames to their
            average precision scores.
    """
    average_precisions = {}

    for query, results in query_results_dict.items():
        document_results = results[0].copy()

        if query in document_results:
            document_results.remove(query)

        num_correct_results = 0
        precision_sum = 0
        for i, doc_result in enumerate(document_results):
            if num_results_to_consider is not None:
                if i >= num_results_to_consider: break

            if doc_result in gold_documents_for_queries_dict[query]:
                num_correct_results += 1
                precision_sum += num_correct_results/(i+1)

        if num_correct_results == 0:
            average_precisions[query] = 0
        else:
            average_precisions[query] = precision_sum/num_correct_results

    return sum(average_precisions.values())/len(average_precisions), average_precisions

def calculate_stats_per_label(data_dir, document_file, query_files, query_results_dict, 
                            gold_documents_for_queries_dict, language):
    """Calculate statistics per label for the given query results. Useful for graphing 
    and analysis.

    Args:
        data_dir (str): directory with all data files
        document_file (str): file with all documents listed
        query_files (str): file with all queries listed
        query_results_dict (dict{str \: tuple(list[str], list[float])}): a dictionary with query 
            basenames as keys, and as values: a tuple with 2 lists: an ordered list of document
            basenames and an ordered list of corresponding document similarities to the query.
            Both lists ordered by similarity.
        gold_documents_for_queries_dict (dict{str \: list[str]}): dictionary with query basenames
            as keys and list of document basenames as values. The documents are those that are
            correct matches for the query.
        language (str): language of the data. Either "banjara" or "tamil"

    Raises:
        NotImplementedError: function is not implemented for Tamil

    Returns:
        tuple(dict{str \: int}, dict{str \: float}, dict{str \: int}, dict{str \: float}
            , dict{str \: float}, dict{str \: float}): tuple of dictionaries with labels as keys and
            the following as values: number of queries with a top 5 match, percent of queries with
            a top 5 match, number of queries with a top 10 match, percent of queries with a top 10 match,
            MAP at 5, MAP overall.
    """
    if language == "tamil":
        raise NotImplementedError("This function is not implemented for Tamil.")
    labels_fnames, _ = banjara_get_data_dicts(data_dir, 0, 10000, query_files, document_file)
    labels_query_results_dict = {}

    for label, fnames in labels_fnames.items():
        query_results_dict_for_label = {}
        for fname in fnames:
            if fname.endswith(".wav"):
                fname = fname.replace(".wav", "")
                if fname in query_results_dict.keys():
                    query_results_dict_for_label[fname] = query_results_dict[fname]
        labels_query_results_dict[label] = query_results_dict_for_label

    labels_top_5 = {}
    labels_top_5_percent = {}
    labels_top_10 = {}
    labels_top_10_percent = {}
    labels_top_5_map = {}
    labels_all_map = {}

    for label, query_results_dict_for_label in labels_query_results_dict.items():
        num_queries_top_5, num_queries_top_10 = \
            calculate_correct_results(query_results_dict_for_label, gold_documents_for_queries_dict)
        map_at_5, _ = calculate_mean_average_precision(
            query_results_dict_for_label, gold_documents_for_queries_dict, num_results_to_consider=5)
        map_all, _ = calculate_mean_average_precision(
            query_results_dict_for_label, gold_documents_for_queries_dict, num_results_to_consider=None)

        labels_top_5[label] = num_queries_top_5
        labels_top_5_percent[label] = round(num_queries_top_5/len(query_results_dict_for_label),3) * 100
        labels_top_10[label] = num_queries_top_10
        labels_top_10_percent[label] = round(num_queries_top_10/len(query_results_dict_for_label),3) * 100
        labels_top_5_map[label] = round(map_at_5, 3)
        labels_all_map[label] = round(map_all, 3)
    
    return labels_top_5, labels_top_5_percent, labels_top_10, labels_top_10_percent, labels_top_5_map, labels_all_map

    
if __name__ == "__main__":

    args = sys.argv

    if len(args) > 1:
        language = args[1]
        layer = int(args[2])
        min_phone_seq_length = int(args[3])
        max_phone_seq_length = int(args[4])
        results_dir_folder = args[5]
        model_type = args[6]
        if len(args) > 7:
            window_size_ms = args[7]  # in milliseconds
            stride_ms = args[8]  # in milliseconds
        else:
            window_size_ms = None  # in milliseconds
            stride_ms = None  # in milliseconds
    else:
        language = "banjara"
        layer = 9
        min_phone_seq_length = 5
        max_phone_seq_length = 14
        results_dir_folder = f"{layer}/tamil_train_3_9_acc/lr_1e-4_tmp_0.07_acc_1000_bs_5_5_14_epoch_0"
        model_type = "awe"
        window_size_ms = None  # in milliseconds
        stride_ms = None  # in milliseconds

    training_seq_lengths = "3-9"
    results_dir_prefix = f"data/{language}/results/{model_type}"
    limit = None  # limit the number of queries to consider
    results_limit = None  # results limit that was used when saving the results, set to None if no limit was used

    graph_results = False
    save_figures = False

    if results_limit is None:
        results_limit = "all"

    # reference file relating queries to their correct documents
    if language == "tamil":
        reference_file = f"data/{language}/analysis/ref_of_queries_in_docs.txt"
    elif language == "banjara":
        reference_file = f"data/{language}/analysis/ref_of_queries_in_docs_nq_99_nd_288.txt"
        documents_file = f"data/{language}/analysis/all_documents.txt"
        queries_file = f"data/{language}/analysis/all_queries.txt"
        all_data_dir = f"data/{language}/banjara_data"

    write_to_file = False
    overall_output_csv_file = f"data/{language}/results/overall_results.csv"

    create_analysis_file = False
    query_audio_dir = f"data/{language}/queries"
    document_audio_dir = f"data/{language}/documents"
    banjara_label_dir = f"data/banjara/banjara_data"
    tamil_label_file = f"data/tamil/analysis/ref_of_queries_in_docs.txt"

    print((f"Results For:\n{language}\nLayer {layer}\nNum results to consider: {results_limit}"
            f"\nInference phone seq lengths: {min_phone_seq_length}-{max_phone_seq_length}"
            f"\nTraining phone seq lengths: {training_seq_lengths}"))

    if "raw_hubert" in results_dir_prefix:
        if window_size_ms is None:
            results_dir = f"{results_dir_prefix}/{layer}/raw_multiple_q_vecs" 
        else:
            _, _, results_dir = \
            qds.get_embedding_and_results_dir("", "", results_dir_prefix, "mean", 
                                            layer, window_size_ms, stride_ms, True)
    elif "awe" or "rec" in results_dir_prefix:
        results_dir = \
            f"{results_dir_prefix}/{results_dir_folder}" 

    results_file = f"{results_dir}/results_{results_limit}.txt"
    print(f"Results for: {results_file}")
    
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Results directory {results_dir} does not exist.")
    
    query_results_dict = get_query_results_dict(results_file, limit=limit)
    num_queries_best_match_itself, num_queries_that_match_with_itself_in_top_10 = \
        count_if_query_matches_itself(query_results_dict)
    
    gold_documents_for_queries_dict = extract_gold_labels_for_queries(reference_file, language)
    num_queries_top_5, num_queries_top_10 = \
        calculate_correct_results(query_results_dict, gold_documents_for_queries_dict)
    
    map_at_5, avg_precision_dict_at_5 = calculate_mean_average_precision(
                                    query_results_dict, gold_documents_for_queries_dict,
                                      num_results_to_consider=5)

    map_all, avg_precision_dict_all = calculate_mean_average_precision(
                                    query_results_dict, gold_documents_for_queries_dict,
                                      num_results_to_consider=None)

    if language == "banjara" and graph_results:
        labels_top5, labels_top5_percent, labels_top10, labels_top10_percent, labels_map_at_5, labels_map_all = \
            calculate_stats_per_label(all_data_dir, documents_file, queries_file, query_results_dict, 
                                    gold_documents_for_queries_dict, language)

        banjara_graph_count_order_bar_chart(labels_top5, "Label", "Number of queries with a top 5 match",
                                            all_data_dir, queries_file, documents_file,
                                                f"figures/labels_top5.png", save_figures)

        banjara_graph_count_order_bar_chart(labels_top5_percent, "Label", 
                                            "Percent of queries with a top 5 match",
                                            all_data_dir, queries_file, documents_file,
                                                f"figures/labels_top5_percent.png", save_figures)

        banjara_graph_count_order_bar_chart(labels_map_at_5, "Label", 
                                            "MAP at 5",
                                            all_data_dir, queries_file, documents_file,
                                                f"figures/labels_map_5.png", save_figures)

        banjara_graph_count_order_bar_chart(labels_map_all, "Label", 
                                            "MAP all",
                                            all_data_dir, queries_file, documents_file,
                                                f"figures/labels_map_all.png", save_figures)
    
    if create_analysis_file:
        analysis_file = f"{results_dir}/results_analysis.txt"
        write_analysis_file(analysis_file, query_results_dict, gold_documents_for_queries_dict,
                            query_audio_dir, document_audio_dir, language, avg_precision_dict_at_5,
                            avg_precision_dict_all, banjara_label_dir, tamil_label_file)

    num_queries = len(query_results_dict)
    percent_queries_that_match_with_itself_in_top_10 = round(num_queries_that_match_with_itself_in_top_10*100/num_queries)
    percent_queries_that_best_match_with_itself = round(num_queries_best_match_itself*100/num_queries)
    percent_queries_top_5 = round(num_queries_top_5*100/num_queries)
    percent_queries_top_10 = round(num_queries_top_10*100/num_queries)
    map_at_5_str = f"{map_at_5:.3f}"
    map_all_str = f"{map_all:.3f}"
        
    if write_to_file:
        try:
            with open(overall_output_csv_file, "r") as f:
                reader = csv.reader(f, delimiter=",")
                rows = list(reader)
        except:
            rows = []
        
        with open(overall_output_csv_file, "a+") as f:
            writer = csv.writer(f)
            if not rows:
                writer.writerow(("layer num_queries %_best_match_itself "
                                "%_top_5 %_top_10 map_at_5 map_all num_best_match_itself "
                                "num_top_5 num_top_10").split(" "))
            row_to_write = [layer, num_queries, percent_queries_that_best_match_with_itself, 
                            percent_queries_top_5, percent_queries_top_10, map_at_5_str, map_all_str, 
                            num_queries_best_match_itself, num_queries_top_5, num_queries_top_10]
            row_to_write_str = [str(x) for x in row_to_write]
            if row_to_write_str not in rows:
                writer.writerow(row_to_write)

    print("Total number of queries:", num_queries)
    print("Number of queries whose best match is its own document:", 
          f"{num_queries_best_match_itself};",
          f"{percent_queries_that_best_match_with_itself:.1f} %")
    print("Number of queries whose own document is in the top 10 matches:", 
          f"{num_queries_that_match_with_itself_in_top_10};", 
          f"{percent_queries_that_match_with_itself_in_top_10:.1f} %")
    print(f"Number of queries with at least one correct result in the top 5 matches:",
          f"{num_queries_top_5};", f"{percent_queries_top_5:.1f} %")
    print(f"Number of queries with at least one correct result in the top 10 matches:",
          f"{num_queries_top_10};", f"{percent_queries_top_10:.1f} %")
    print(f"Mean Average Precision at 5: {map_at_5_str}")
    print(f"Mean Average Precision Overall: {map_all_str}")
