"""Examine various aspects of the datasets, including the duration of the data, the duration of voice activity,
and the number of documents in the search corpus."""
import os
import re
from collections import defaultdict

from utils.get_document_lengths import get_wav_file_length

def clean_string(string):
    """Removes unwanted characters from string."""
    string = string.replace(".pkl", "")
    string = string.replace(".wav", "")
    string = string.replace("q_", "")
    string = string.replace("\n", "")
    string = string.strip()
    return string

def extract_gold_labels_for_queries_tamil(reference_file):
    """Given a reference file with queries, their corresponding labels and correct documents,
    return two dictionaries: one with queries as keys and a list of their corresponding correct
    documents as values, another with queries as keys and their corresponding labels as values.
    Each line in the reference file is assumed to have the following format: 
    "[query] [tab] [label] [tab] [correct_document_1] [tab] [correct_document_2]..."
    where the spaces are just for readability and [tab] is a tab character. 
    Note that this function is specific to our setup with the tamil dataset.

    Args:
        reference_file (str): path to reference file

    Returns:
        tuple(dict{str \: list[str]}, dict{str \: str}): first dictionary has queries as keys and
            lists of correct documents as values, second dictionary has queries as keys and their
            corresponding labels as values.
    """
    gold_documents_for_queries_dict = {}
    query_labels = {}

    with open(reference_file, "r") as f:
        for line in f:
            split_line = line.split("\t")
            query = split_line[0]
            query = clean_string(query)
            query_labels[query] = split_line[1]
            documents = split_line[2:-1]
            gold_documents_for_queries_dict[query] = documents
    
    return gold_documents_for_queries_dict, query_labels

def extract_gold_labels_for_queries_banjara(reference_file):
    """Given a reference file with queries and their corresponding correct documents,
    return a dictionary with queries as keys and a list of their corresponding correct
    documents as values. Each line in the reference file is assumed to have the following format:
    "[query] [tab] [correct_document_1] [tab] [correct_document_2]..."
    where the spaces are just for readability and [tab] is a tab character.
    This function is specific to our setup with the banjara dataset.

    Args:
        reference_file (str): path to the reference file

    Returns:
        dict{str \: list[str]}: dictionary with queries as keys and a list of their 
            corresponding correct documents as values
    """
    gold_documents_for_queries_dict = {}

    with open(reference_file, "r") as f:
        for line in f:
            split_line = line.split("\t")
            query = split_line[0]
            query = clean_string(query)
            documents = split_line[1:-1]
            gold_documents_for_queries_dict[query] = documents
    
    return gold_documents_for_queries_dict

def get_all_data_files(all_data_dir):
    """Given a directory, return a set of all the files
    in the directory that end with ".wav".

    Args:
        all_data_dir (str): path to directory

    Returns:
        set{str}: set of basenames of all files in the directory that end with ".wav"
    """
    files = os.listdir(all_data_dir)
    files_to_remove = []

    for i, file in enumerate(files):
        if file.endswith(".wav"):
            files[i] = clean_string(file)
        else:
            files_to_remove.append(file)
    
    for file in files_to_remove:
        files.remove(file)
    
    return set(files)

def read_documents_file(file_path):
    """Given a documents file, return a set of basenames of all documents
    in the file. This file should be formatted with one document basename per line.

    Args:
        file_path (str): path to the documents file

    Returns:
        set{str}: set of basenames of all documents in the file
    """
    all_docs = set()
    with open(file_path) as f:
        for row in f:
            all_docs.add(clean_string(row))
    return all_docs

def read_phone_timings_file(file_path):
    """Reads phone timings file and produces a defaultdict with file basenames as keys and 
    a dictionary with the corresponding phones in the file and their corresponding start times
    and durations as values. Files that only contain silence are ignored and are not included
    in the output dictionary. Each line in the phone timings file is assumed to have the following format:
    "[doc_basename] 1 [start_time] [duration] [phone]", where the spaces are just for readability.
    
    Args:
        file_path (str): path of the phone timings file

    Returns:
        dict{str \: dict{"phones" \: list[str], "start_times" \: list[float], "durations" \: list[float]}},
            default dictionary with file basenames as keys and a dictionary with the corresponding phones in the file,
            and their corresponding start times and durations as values. 
    """
    files_phones_dict = defaultdict(lambda: {"phones": [], "start_times": [], "durations": []})

    prev_doc_name = ""
    prev_doc_has_non_sil_phones = False
    sil_phones = set(["sil", "sp", "spn"])

    with open(file_path) as f:
        for row in f:
            split_string = row.split(" ")
            doc_name = re.sub(r"^[a-z]+_", "", split_string[0])
            # doc_name = split_string[0].replace("tamil_", "")

            if prev_doc_name != doc_name and prev_doc_name != "":
                if not prev_doc_has_non_sil_phones:
                    files_phones_dict.pop(prev_doc_name)
                prev_doc_has_non_sil_phones = False

            one_value = split_string[1]
            if one_value != "1":
                print(f"Value is not 1 for {doc_name}")
            
            start_time = float(split_string[2])
            duration = float(split_string[3])
            phone = split_string[4].replace("\n", "")

            if phone not in sil_phones:
                prev_doc_has_non_sil_phones = True

            files_phones_dict[doc_name]["phones"].append(phone)
            files_phones_dict[doc_name]["start_times"].append(start_time)
            files_phones_dict[doc_name]["durations"].append(duration)

                    
            prev_doc_name = doc_name

    return files_phones_dict

def get_duration_of_docs(document_set, document_dir):
    """Calculate the duration of all the documents in the document set,
    given the directory where the documents are stored.

    Args:
        document_set (set{str}): set of document basenames
        document_dir (str): path of the directory where the documents are stored

    Returns:
        float: duration of all the documents in the document set
    """
    duration = 0
    for document in document_set:
        doc_duration = get_wav_file_length(f"{document_dir}/{document}.wav")
        duration += doc_duration
    
    return duration

def get_voice_activity_duration_of_docs(document_set, files_phones_dict, sil_phones=["sil", "sp", "spn"]):
    """Calculate the duration of voice activity in all the documents in the document set,
    given the dictionary with the phones in the documents and their corresponding start times and durations.

    Args:
        document_set (set{str}): set of document basenames
        files_phones_dict (dict{str \: dict{"phones" \: list[str], "start_times" \: list[float], "durations" \: list[float]}}):
            dictionary with document basenames as keys and a dictionary with the corresponding phones in the document,
            and their corresponding start times and durations as values.
        sil_phones (list, optional): list of silence phones to ignore. Defaults to ["sil", "sp", "spn"].

    Returns:
        float: voice activity duration of all the documents in the document set.
    """
    duration = 0
    for document in document_set:
        phones = files_phones_dict[document]["phones"]
        durations = files_phones_dict[document]["durations"]

        for i, phone in enumerate(phones):
            if phone not in sil_phones:
                duration += durations[i]
    
    return duration


if __name__ == "__main__":
    # dir where all the data is stored, including data that was filtered out after preprocessing
    language = "tamil"
    all_data_dir = f"data/{language}/all_data"
    queries_dir = f"data/{language}/queries"
    analysis_dir = f"data/{language}/analysis"

    # reference file relating queries to their correct documents
    reference_file = f"{analysis_dir}/ref_of_queries_in_docs.txt"
    # reference_file = f"{analysis_dir}/ref_of_queries_in_docs_nq_99_nd_288.txt"

    # file containing all documents in search corpus
    # all_documents_file = f"{analysis_dir}/all_documents_288.txt"
    all_documents_file = f"{analysis_dir}/all_documents.txt"

    phone_timings_file = f"{analysis_dir}/phone_all.ctm"

    training_data_save_file = f"{analysis_dir}/training_data_quarter.txt"
    validation_data_save_file = f"{analysis_dir}/validation_data.txt"

    num_to_sample = 60

    if language == "tamil":
        gold_documents_for_queries_dict, tamil_labels = \
            extract_gold_labels_for_queries_tamil(reference_file)
    elif language == "banjara":
        gold_documents_for_queries_dict = extract_gold_labels_for_queries_banjara(reference_file)
    
    all_data_files_set = get_all_data_files(all_data_dir)

    all_docs_in_search_corpus_set = read_documents_file(all_documents_file)

    files_phones_dict = read_phone_timings_file(phone_timings_file)

    docs_not_in_search_corpus_set = all_data_files_set - all_docs_in_search_corpus_set 

    docs_not_in_search_corpus_set = docs_not_in_search_corpus_set & set(files_phones_dict.keys())

    silence_docs = all_data_files_set - set(files_phones_dict.keys())

    # not_in_search_corpus_duration = get_duration_of_docs(docs_not_in_search_corpus_set, all_data_dir)

    # not_in_search_corpus_voice_activity_duration = \
    #     get_voice_activity_duration_of_docs(docs_not_in_search_corpus_set, files_phones_dict)

    # query_files_set = get_all_data_files(queries_dir)
    # duration = get_duration_of_docs(query_files_set, queries_dir)
    # print(f"Duration of queries: {duration:.0f} s, {duration/60:.1f} m, {duration/3600:.2f} h")
    # average_duration = duration / len(query_files_set)
    # print(f"Average duration of queries: {average_duration:.0f} s, {average_duration/60:.1f} m, {average_duration/3600:.2f} h")

    # all_length = get_duration_of_docs(all_data_files_set, all_data_dir)
    # print(f"Duration of all data: {all_length:.0f} s, {all_length/60:.1f} m, {all_length/3600:.2f} h")
    # print(f"Length of all data set: {len(all_data_files_set)}")

    search_duration = get_duration_of_docs(all_docs_in_search_corpus_set, all_data_dir)
    print(f"Duration of search corpus: {search_duration:.0f} s, {search_duration/60:.1f} m, {search_duration/3600:.2f} h")
    print(f"Length of search corpus set: {len(all_docs_in_search_corpus_set)}")

    # training_files = read_documents_file(training_data_save_file)
    # silent_training_files = training_files & silence_docs
    # training_files_not_silent = training_files - silent_training_files
    # training_duration = get_duration_of_docs(training_files_not_silent, all_data_dir)
    # training_voice_activity_duration = \
    #     get_voice_activity_duration_of_docs(training_files_not_silent, files_phones_dict)

    # print(f"Num docs in training set: {len(training_files)}")
    # print(f"Num docs in training set that are silent: {len(silent_training_files)}")
    # print(f"Num total silent docs: {len(silence_docs)}")

    # sample and remove from search corpus
    # sample_from_corpus = set(random.sample(sorted(all_docs_in_search_corpus_set), num_to_sample))
    # all_docs_in_search_corpus_set.difference_update(sample_from_corpus)
    # search_corpus_duration = get_duration_of_docs(all_docs_in_search_corpus_set, all_data_dir)
    # training_corpus = docs_not_in_search_corpus_set | sample_from_corpus
    # training_duration = get_duration_of_docs(training_corpus, all_data_dir)
    # training_voice_activity_duration = \
    #     get_voice_activity_duration_of_docs(training_corpus, files_phones_dict)
    
    # print((f"Duration of documents that are not in search corpus and have some non silence phones: "
    #        f"{not_in_search_corpus_duration:.0f} s, {not_in_search_corpus_duration/60:.1f} m, "
    #        f"{not_in_search_corpus_duration/3600:.2f} h"))

    # print((f"Duration of voice activity of documents that are not in search corpus: "
    #        f"{not_in_search_corpus_voice_activity_duration:.0f} s, {not_in_search_corpus_voice_activity_duration/60:.1f} m, "
    #        f"{not_in_search_corpus_voice_activity_duration/3600:.2f} h"))

    # print((f"Duration of training documents: "
    #        f"{training_duration:.0f} s, {training_duration/60:.1f} m, "
    #        f"{training_duration/3600:.2f} h"))

    # print((f"Duration of voice activity in training documents: "
    #        f"{training_voice_activity_duration:.0f} s, {training_voice_activity_duration/60:.1f} m, "
    #        f"{training_voice_activity_duration/3600:.2f} h"))

    # print((f"Duration of remaining search documents: "
    #        f"{search_corpus_duration:.0f} s, {search_corpus_duration/60:.1f} m, "
    #        f"{search_corpus_duration/3600:.2f} h"))
        
    # print(f"Num docs remaining in search corpus set: {len(all_docs_in_search_corpus_set)}")




