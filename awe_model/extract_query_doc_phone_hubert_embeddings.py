"""Get AWE representation for queries and documents through splitting the speech signal into phone segments 
and extracting embeddings for each segment. This script is DEPRECATED and should not be used. Splitting 
recordings into speech signal segments produces much worse performance than first embedding the whole
recording using mHuBERT and then splitting into segments. This could be since mHuBERT may struggle with the 
unnatural discontinuities at the edges of cut out segments of the signal."""
import sys
import os
from collections import defaultdict
import time
import pickle as pkl
import csv
import re

from mhubert_model.extract_mHuBERT_embeddings import HubertEmbedder 
from utils.common_functions import read_wav_file, split_list_into_n_parts_and_get_part, make_dir
from utils.examine_datasets import read_phone_timings_file

def read_query_times_file(query_times_file):
    """Read query times file. Useful if queries were cut out of longer recordings.

    Args:
        query_times_file (str): path of the file containing the times of queries
            in their original recordings. Should be formatted as: 
            "<original_file_basename> <tab> <query_keyword> <tab> <start_time> <tab> <end_time>"
            where the spaces are just for readability, <tab> is a tab character and start_time 
            and end_time are in seconds.

    Returns:
        dict{str \: tuple(float, float)}: dictionary mapping file names to start and end times
    """
    query_times_dict = {}
    with open(query_times_file, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for line in reader:
            original_filename, _, start_time, end_time = line
            file = original_filename
            counter = 1
            while file in query_times_dict.keys():
                counter += 1
                file = original_filename + f"_{counter}"
            query_times_dict[file] = (float(start_time), float(end_time))
    return query_times_dict


def get_file_phone_segments_dict(audio_directory, phone_timings_file, query_times_file=None,
                                 sample_rate = 16000, min_phone_seq_length = 3, 
                                 max_phone_seq_length = 9, silence_phones = ["sil", "sp", "spn"]):
    """Create a dictionary mapping file names to phone segments in the files.

    Args:
        audio_directory (str): directory containing audio files.
        phone_timings_file (str): file containing phone timings for all recordings.
        query_times_file (str, optional): File path of query times file. Use if queries were
            cut out of longer recordings, file specifies start/end times of query in original 
            recording. Defaults to None.
        sample_rate (int, optional): sample rate of audio. Defaults to 16000.
        min_phone_seq_length (int, optional): min length (in phones) of phone sequence to consider.
            Defaults to 3.
        max_phone_seq_length (int, optional): max length (in phones) of phone sequence to consider.
            Defaults to 9.
        silence_phones (list, optional): list of phones to ignore. Defaults to ["sil", "sp", "spn"].

    Returns:
        dict{str \: list[list[float]]}: defaultdict(list) mapping file names to a list of phone segments. 
        Phone segments are cut directly out of the speech signal.
    """
    files_phones_dict = read_phone_timings_file(phone_timings_file)
    if query_times_file is not None:
        query_times_dict = read_query_times_file(query_times_file)
    else:
        query_times_dict = None

    file_phone_segments_dict = defaultdict(list)
    files = sorted(os.listdir(audio_directory))
    for file in files:
        speech = read_wav_file(f"{audio_directory}/{file}", sample_rate)
        file = file.replace(".wav", "")
        file = file.replace("q_", "")
        file_key = re.sub(r"_\d+$", "", file)
        phones = files_phones_dict[file_key]["phones"]
        starts = files_phones_dict[file_key]["start_times"]
        durations = files_phones_dict[file_key]["durations"]

        if query_times_dict is not None:
            start_time, end_time = query_times_dict[file]
            moved_starts = []
            moved_durations = []
            moved_phones = []
            for i, start in enumerate(starts):
                if start >= start_time and start < end_time:
                    moved_starts.append(start - start_time)
                    moved_durations.append(durations[i])
                    moved_phones.append(phones[i])
            
            starts = moved_starts
            durations = moved_durations
            phones = moved_phones
        
        for length in range(min_phone_seq_length, max_phone_seq_length+1):
            for i in range(len(phones)):
                if i+length > len(phones):
                    break
                phone_seq = phones[i:i+length]
                if any(phone in silence_phones for phone in phone_seq):
                    continue
                start_time = int(starts[i] * sample_rate)
                durations_sum = int(sum(durations[i:i+length]) * sample_rate)

                file_phone_segments_dict[file].append(
                    speech[start_time:start_time+durations_sum])

    return file_phone_segments_dict

if __name__ == "__main__":
    top_level_dir = "data/tamil/"
    audio_folder = "queries"
    audio_dir = f"{top_level_dir}/{audio_folder}"
    phone_timings_file = f"{top_level_dir}/analysis/phone_all.ctm"

    # set to none if queries are not cut out of longer documents
    query_times_file = f"{top_level_dir}/analysis/queries_times.txt"
    # query_times_file = None

    t1 = time.perf_counter()
    file_phone_segments_dict = get_file_phone_segments_dict(audio_dir, phone_timings_file, 
                                                            query_times_file)
    t2 = time.perf_counter()
    print(f"TIME TO EXTRACT PHONE SEGMENTS: {t2-t1:.2f} s")

    layers = [8,9]
    device = "cpu"
    top_level_embedding_dir = f"{top_level_dir}/embeddings"
    folder = f"{audio_folder}_phone_segments"

    n_parts = 1
    if n_parts > 1:
        part = sys.argv[1]

    t1 = time.perf_counter()
    hubert = HubertEmbedder(device, hidden_layers=layers)
    t2 = time.perf_counter()
    print(f"TIME TO LOAD HUBERT: {t2-t1:.2f} s")
    print(f"NUMBER OF CPU CORES: {os.cpu_count()}")

    t1 = time.perf_counter()
    print(f"Generating embeddings for {folder}")

    for lay in layers:
        embedding_directory = f"{top_level_embedding_dir}/{folder}/{lay}/raw"
        print(f"Embedding directory for layer {lay}: {embedding_directory}")
        make_dir(embedding_directory)

    dataset_length = len(file_phone_segments_dict)

    files = list(file_phone_segments_dict.keys())
    file_part = split_list_into_n_parts_and_get_part(files, n_parts, part)

    for idxs, file in enumerate(file_part):
        multiple_speech = file_phone_segments_dict[file]
        hidden_states_for_file = [[] for _ in range(len(layers))] 
        print(file)

        for speech in multiple_speech:
            hidden_states = hubert.embed_speech(speech)
            for i in range(len(layers)):
                hidden_states_for_file[i].append(hidden_states[i][0])

        percentage = (idxs + 1)/dataset_length * 100
        print(f"{percentage:.2f}% done")
        t = time.perf_counter()
        print(f"Time: {t - t1:.2f} s")

        for lay_idx, lay in enumerate(layers):
            embedding_directory = f"{top_level_embedding_dir}/{folder}/{lay}/raw"
            file = file.replace(" ", "_")
            embedding_fname = os.path.join(embedding_directory, f"{file}.pkl")
            with open(embedding_fname, "wb") as f:
                pkl.dump(hidden_states_for_file[lay_idx], f)

    t2 = time.perf_counter()
    print(f"TIME TO EXTRACT EMBEDDINGS: {t2-t1:.2f} s")