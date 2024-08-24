import sys
import os
from collections import defaultdict
import time
import pickle as pkl
import soundfile as sf
from mhubert_model.extract_mHuBERT_embeddings import HubertEmbedder 
from utils.common_functions import split_list_into_n_parts_and_get_part, make_dir, read_wav_file
from utils.split_data_test_train import read_phone_timings_file


def get_matching_phone_segments_dict(audio_directory, phone_timings_file, sample_rate = 16000, 
                min_phone_seq_length = 2, max_phone_seq_length = 5, silence_phones = ["sil", "sp", "spn"]):
    files_phones_dict = read_phone_timings_file(phone_timings_file)

    matching_phone_segments_dict = defaultdict(list)
    files = sorted(os.listdir(audio_directory))
    for file in files:
        speech = read_wav_file(f"{audio_directory}/{file}", sample_rate)
        file = file.replace(".wav", "")
        phones = files_phones_dict[file]["phones"]
        starts = files_phones_dict[file]["start_times"]
        durations = files_phones_dict[file]["durations"]
        
        for length in range(min_phone_seq_length, max_phone_seq_length+1):
            for i in range(len(phones)):
                if i+length > len(phones):
                    break
                phone_seq = phones[i:i+length]
                if any(phone in silence_phones for phone in phone_seq):
                    continue
                start_time = int(starts[i] * sample_rate)
                durations_sum = int(sum(durations[i:i+length]) * sample_rate)

                phone_seq_str = " ".join(phone_seq)
                matching_phone_segments_dict[phone_seq_str].append(
                    speech[start_time:start_time+durations_sum])

    return matching_phone_segments_dict

if __name__ == "__main__":
    top_level_dir = "data/tamil/"
    audio_dir = f"{top_level_dir}/training_data"
    phone_timings_file = f"{top_level_dir}/analysis/phone_all.ctm"
    t1 = time.perf_counter()
    matching_phone_segments_dict = get_matching_phone_segments_dict(audio_dir, phone_timings_file)
    t2 = time.perf_counter()
    print(f"TIME TO EXTRACT PHONE SEGMENTS: {t2-t1:.2f} s")

    layers = [8,9]
    device = "cuda"
    top_level_embedding_dir = f"{top_level_dir}/embeddings"
    folder = "training_data"

    n_parts = 10
    part = sys.argv[1]

    t1 = time.perf_counter()
    hubert = HubertEmbedder(device, hidden_layers=layers)
    t2 = time.perf_counter()
    print(f"TIME TO LOAD HUBERT: {t2-t1:.2f} s")
    print(f"NUMBER OF CPU CORES: {os.cpu_count()}")

    t1 = time.perf_counter()
    print(f"Generating embeddings for {folder}")
    audio_directory = f"{top_level_dir}/{folder}"

    for lay in layers:
        embedding_directory = f"{top_level_embedding_dir}/{folder}/{lay}/raw"
        make_dir(embedding_directory)

    dataset_length = len(matching_phone_segments_dict)

    phone_seqs = list(matching_phone_segments_dict.keys())
    phone_seq_part = split_list_into_n_parts_and_get_part(phone_seqs, n_parts, part)

    for idxs, phone_seq in enumerate(phone_seq_part):
        multiple_speech = matching_phone_segments_dict[phone_seq]
        hidden_states_for_phone_seq = [[] for _ in range(len(layers))] 
        print(phone_seq)

        for speech in multiple_speech:
            hidden_states = hubert.embed_speech(speech)
            for i in range(len(layers)):
                hidden_states_for_phone_seq[i].append(hidden_states[i])

        percentage = (idxs + 1)/dataset_length * 100
        print(f"{percentage:.2f}% done")
        t = time.perf_counter()
        print(f"Time: {t - t1:.2f} s")

        for lay_idx, lay in enumerate(layers):
            embedding_directory = f"{top_level_embedding_dir}/{folder}/{lay}/raw"
            phone_seq = phone_seq.replace(" ", "_")
            embedding_fname = os.path.join(embedding_directory, f"{phone_seq}.pkl")
            with open(embedding_fname, "wb") as f:
                pkl.dump(hidden_states_for_phone_seq[lay_idx], f)

    t2 = time.perf_counter()
    print(f"TIME TO EXTRACT EMBEDDINGS: {t2-t1:.2f} s")