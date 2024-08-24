import os
from collections import defaultdict
import random
import soundfile as sf
from utils.common_functions import get_wav_file_length

def read_label(label_fname):
    with open(label_fname, "r") as f:
        lines = f.readlines()
    label = lines[0].strip()
    return label

if __name__ == "__main__":
    data_dir = "data/banjara/banjara_data"
    analysis_dir = "data/banjara/analysis"
    files = sorted(os.listdir(data_dir))
    labels_fnames = defaultdict(list)
    fnames_labels = {}
    approx_num_queries = 99
    duration_min = 2.0
    duration_max = 180

    # allowed_labels = ["millet", "corn", "chickpea", "tur dal", "wheat", "cotton", "groundnut", "parsley",
                    #   "onion", "eggplant", "beans", "lemon", "radish", "pomegranate", "taro", "papaya",
                    #   "guava", "stove", "dried dal"]
    label_duration_dict = defaultdict(float)
    
    for i, fname in enumerate(files):
        if fname.endswith(".wav"):
            duration = get_wav_file_length(os.path.join(data_dir, fname))
            if duration < duration_min or duration > duration_max:
                continue
            label_fname = fname.replace(".wav", ".label")
            label = read_label(os.path.join(data_dir, label_fname))
            label_duration_dict[label] += duration
            labels_fnames[label].append(fname)
    
    sorted_label_duration_dict = sorted(label_duration_dict.items(), key=lambda x: x[1], reverse=True)
    print(len(sorted_label_duration_dict))
    for key, value in sorted_label_duration_dict:
        print(f"{key}: {value:.1f} s", end=", ")
    
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
    
    queries_list = []
    for label, fnames in labels_fnames.items():
        num_to_sample = round((len(fnames)/num_of_files) * approx_num_queries)
        sampled_list = random.sample(fnames, num_to_sample)
        queries_list.extend(sampled_list)
    
    print(f"Number of queries: {len(queries_list)}")
    print(f"Number of documents: {num_of_files}")

    # with open(os.path.join(analysis_dir, "all_queries.txt"), "w") as f:
    #     for fname in queries_list:
    #         fname = fname.replace(".wav", "")
    #         f.write(f"{fname}\n")

    # with open(os.path.join(analysis_dir, "all_documents.txt"), "w") as f:
    #     for fname in fnames_labels.keys():
    #         fname = fname.replace(".wav", "")
    #         f.write(f"{fname}\n")

    ls = set()
    with open(os.path.join(analysis_dir, "ref_of_queries_in_docs.txt"), "w") as f:
        for q_fname in queries_list:
            label = fnames_labels[q_fname]
            ls.add(label)
            document_names = labels_fnames[label]
            document_names = filter(lambda x: x != q_fname, document_names)
            query_name = "q_" + q_fname.replace(".wav", "")
            # f.write(f"{query_name}\t")
            # for fname in document_names:
            #     fname = fname.replace(".wav", "")
            #     f.write(f"{fname}\t")
            # f.write("\n")
    print(ls)
    print(len(ls))
    # print(f"Number of unique labels in query set: {len(ls)} out of {len(labels_fnames.keys())} labels in the dataset.")