"""Use to create queries and documents for the Banjara dataset."""
import os
import random
from collections import defaultdict

from utils.common_functions import get_wav_file_length
from utils.examine_datasets import read_documents_file

def read_label(label_fname):
    """Read label from label file.

    Args:
        label_fname (str): path to label file

    Returns:
        str: label
    """
    with open(label_fname, "r") as f:
        lines = f.readlines()
    label = lines[0].strip()
    return label

def filter_data_dict(files, documents_file=None, queries_file=None):
    """Removes entries from files that are not in documents_file or queries_file. files must be
    a list of file names.

    Args:
        files (list[str]): list of file names
        documents_file (str): path of file containing document file names
        queries_file (str): path of file containing query file names

    Returns:
        list[str]: list of file names that are in documents_file or queries_file, includes label
            files and wav files
    """
    if documents_file is None and queries_file is None:
        print(f"Neither documents_file nor queries_file provided for filtering. Returning all files.")
        return files
    if documents_file is None:
        documents_file = queries_file
    elif queries_file is None:
        queries_file = documents_file

    documents = read_documents_file(documents_file)
    queries = read_documents_file(queries_file)

    filtered_files = []
    for f in files:
        basename = f.replace(".wav", "")
        basename = basename.replace(".label", "")
        if basename in documents or basename in queries:
            filtered_files.append(f)

    return filtered_files

def banjara_get_data_dicts(all_data_dir, duration_min, duration_max, queries_file=None, documents_file=None):
    """Get data dictionaries for banjara dataset. First dictionary maps labels to lists of filenames, where
    the list of filenames are the audio files that have the corresponding label. Second dictionary maps 
    labels to total duration of audio files for each label.
    Optionally filters out files that are not in queries_file or documents_file. Set queries_file and
    documents_file to None to include all files.

    Args:
        all_data_dir (str): directory of all wav and label files
        duration_min (float): minimum duration of audio file
        duration_max (float): maximum duration of audio file
        queries_file (str): path of file containing query file names
        documents_file (str): path of file containing document file names

    Returns:
        tuple(dict{str \: list[str]}, dict{str \: float}): two data dictionaries for the banjara dataset
    """
    files = sorted(os.listdir(all_data_dir))
    files = filter_data_dict(files, documents_file, queries_file)
    labels_fnames = defaultdict(list)
    label_duration_dict = defaultdict(float)
    
    for i, fname in enumerate(files):
        if fname.endswith(".wav"):
            duration = get_wav_file_length(os.path.join(all_data_dir, fname))
            if duration < duration_min or duration > duration_max:
                continue
            label_fname = fname.replace(".wav", ".label")
            label = read_label(os.path.join(all_data_dir, label_fname))
            label_duration_dict[label] += duration
            labels_fnames[label].append(fname)

    return labels_fnames, label_duration_dict
    

if __name__ == "__main__":
    data_dir = "data/banjara/banjara_data"
    analysis_dir = "data/banjara/analysis"
    fnames_labels = {}
    approx_num_queries = 99
    duration_min = 0
    duration_max = 10000

    # allowed_labels = ["millet", "corn", "chickpea", "tur dal", "wheat", "cotton", "groundnut", "parsley",
                    #   "onion", "eggplant", "beans", "lemon", "radish", "pomegranate", "taro", "papaya",
                    #   "guava", "stove", "dried dal"]
    
    labels_fnames, label_duration_dict = banjara_get_data_dicts(data_dir, duration_min, duration_max)
    
    # remove classes with only one file
    labels_to_remove = []
    for label, fnames in labels_fnames.items():
        if len(labels_fnames[label]) < 2:
            labels_to_remove.append(label)
        else:
            for fname in fnames:
                fnames_labels[fname] = label
    
    for label in labels_to_remove:
        del labels_fnames[label]
    
    num_of_files = len(fnames_labels.keys())
    
    # randomly sample queries, proportional to number of documents in each class
    # ensure that each class has at least one query
    queries_list = []
    for label, fnames in labels_fnames.items():
        num_to_sample = round((len(fnames)/num_of_files) * approx_num_queries)
        sampled_list = random.sample(fnames, num_to_sample)
        queries_list.extend(sampled_list)
    
    print(f"Number of queries: {len(queries_list)}")
    print(f"Number of documents: {num_of_files}")

    # write queries and documents to files
    with open(os.path.join(analysis_dir, "all_queries.txt"), "w") as f:
        for fname in queries_list:
            fname = fname.replace(".wav", "")
            f.write(f"{fname}\n")

    with open(os.path.join(analysis_dir, "all_documents.txt"), "w") as f:
        for fname in fnames_labels.keys():
            fname = fname.replace(".wav", "")
            f.write(f"{fname}\n")

    ls = set()
    with open(os.path.join(analysis_dir, "ref_of_queries_in_docs.txt"), "w") as f:
        for q_fname in queries_list:
            label = fnames_labels[q_fname]
            ls.add(label)
            document_names = labels_fnames[label]
            document_names = filter(lambda x: x != q_fname, document_names)
            query_name = "q_" + q_fname.replace(".wav", "")
            f.write(f"{query_name}\t")
            for fname in document_names:
                fname = fname.replace(".wav", "")
                f.write(f"{fname}\t")
            f.write("\n")
    # print(ls)
    # print(len(ls))
    # print(f"Number of unique labels in query set: {len(ls)} out of {len(labels_fnames.keys())} labels in the dataset.")