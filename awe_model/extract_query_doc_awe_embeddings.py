import time
import os
import sys
import pickle as pkl
import re
import math
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from awe_model.model import SSEmodel
from awe_model.extract_query_doc_phone_hubert_embeddings import read_query_times_file
from utils.common_functions import make_dir, split_list_into_n_parts_and_get_part, split_list_into_n_parts, parse_boolean_input
from utils.split_data_test_train import read_phone_timings_file

class PhoneSplitter:
    def __init__(self, phone_timings_file, query_times_file=None,
                  min_phone_seq_length = 2, max_phone_seq_length = 5,
                    silence_phones = ["sil", "sp", "spn"]):

        self.files_phones_dict = read_phone_timings_file(phone_timings_file)

        if query_times_file is not None:
            self.query_times_dict = read_query_times_file(query_times_file)
        else:
            self.query_times_dict = None
        
        self.hubert_sampling_rate = 50  # each vector is 20 ms
        self.min_phone_seq_length = min_phone_seq_length
        self.max_phone_seq_length = max_phone_seq_length
        self.silence_phones = silence_phones

    def get_start_duration_phones_for_file(self, file):    
        file = file.replace(".pkl", "")
        file = file.replace(".wav", "")
        file = file.replace("q_", "")
        file_key = re.sub(r"_\d+$", "", file)
        phones = self.files_phones_dict[file_key]["phones"]
        starts = self.files_phones_dict[file_key]["start_times"]
        durations = self.files_phones_dict[file_key]["durations"]

        if self.query_times_dict is not None:
            start_time, end_time = self.query_times_dict[file]
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
        
        return phones, starts, durations

    def split_embedded_data_into_phones_list(self, embedded_speech, file):

        phones, starts, durations = self.get_start_duration_phones_for_file(file)
        
        output_embedding_list = []
        for length in range(self.min_phone_seq_length, self.max_phone_seq_length+1):
            for i in range(len(phones)):
                if i+length > len(phones):
                    break
                phone_seq = phones[i:i+length]
                if any(phone in self.silence_phones for phone in phone_seq):
                    continue
                start_time = math.floor(starts[i] * self.hubert_sampling_rate)
                durations_sum = math.ceil(sum(durations[i:i+length]) * self.hubert_sampling_rate)

                output_embedding_list.append(embedded_speech[start_time:start_time+durations_sum])
        
        if output_embedding_list == []:
            print(f"No phone sequences found for {file}")
            speech_length = embedded_speech.shape[0]
            if speech_length > 1000:
                n = math.ceil(speech_length / 1000)
                output_embedding_list = split_list_into_n_parts(embedded_speech, n)
            else:
                output_embedding_list = [embedded_speech]

        return output_embedding_list
    
    def split_embedded_data_into_list(self, embedded_speech, file):
        return self.split_embedded_data_into_phones_list(embedded_speech, file)

    def split_embedded_data_into_phones_and_append_to_dict(self, embedded_speech, file, phone_dict,
                                                           perturb_sequences=False, 
                                                           max_one_sided_perturb_amount=0.2):
        """split embedded speech into segments of 2 to 5 phones and append to phone_dict. phone_dict
            is a defaultdict(list) where the key is a string of the phones and the value is a list of the
            embedded speech segments.

        Args:
            embedded_speech (tensor): the embedded speech tensor for the file of shape [num_hubert_frames, 768]
            file (string): filename of the embedded file either .pkl or .wav
            phone_dict (defaultdict(list)): defaultdict of list of embedded speech segments for 
                each phone sequence
            perturb_sequences (bool, optional): whether to perturb the sequences. Defaults to False.
                if true, edges of phone sequences are perturbed by a random amount.
            max_one_sided_perturb_amount (float, optional): maximum amount to perturb 
                the edges of the phone sequences. Is measured as a fraction of the phone sequence length.
                Defaults to 0.2.
        """

        phones, starts, durations = self.get_start_duration_phones_for_file(file)

        if perturb_sequences:
            padding_offset = math.ceil(max_one_sided_perturb_amount * embedded_speech.shape[0])
            embedded_speech = \
                F.pad(embedded_speech, (0,0, padding_offset, padding_offset), 'constant', 0)
        
        for length in range(self.min_phone_seq_length, self.max_phone_seq_length+1):
            for i in range(len(phones)):
                if i+length > len(phones):
                    break
                phone_seq = phones[i:i+length]
                if any(phone in self.silence_phones for phone in phone_seq):
                    continue
                start_time = math.floor(starts[i] * self.hubert_sampling_rate)
                durations_sum = math.ceil(sum(durations[i:i+length]) * self.hubert_sampling_rate)

                phone_seq = " ".join(phone_seq)

                if perturb_sequences:
                    max_perturb_amount = round(max_one_sided_perturb_amount * durations_sum)
                    perturb_amounts = \
                        torch.randint(-max_perturb_amount, max_perturb_amount+1, (2,))
                    speech_segment = \
                        embedded_speech[start_time + padding_offset + perturb_amounts[0]:
                                        start_time + durations_sum + padding_offset + perturb_amounts[1]
                                        ].unsqueeze(0)
                else:
                    speech_segment = embedded_speech[start_time:start_time+durations_sum].unsqueeze(0)
                
                phone_dict[phone_seq].append(speech_segment)

class WindowSplitter:
    def __init__(self, phone_length_secs=0.08, overlap_fraction=0.5, min_phone_seq_length=3, 
                 max_phone_seq_length=9):
        """initialise the WindowSplitter class

        Args:
            phone_length_secs (float, optional): Average length of a phone, in seconds. Defaults to 0.08.
            overlap_fraction (float, optional): Amount to overlap windows, 0.5=50%. Defaults to 0.5.
            min_phone_seq_length (int, optional): Min length of phone sequence/window. Defaults to 3.
            max_phone_seq_length (int, optional): Max length of phone sequence/window. Defaults to 9.
        """
        self.phone_length_secs = phone_length_secs
        self.overlap_fraction = overlap_fraction
        self.min_phone_seq_length = min_phone_seq_length
        self.max_phone_seq_length = max_phone_seq_length
        self.hubert_sampling_rate = 50  # each vector is 20 ms

    def split_embedded_data_into_windows_list(self, embedded_speech):
        speech_length = embedded_speech.shape[0]
        windows = []

        for num_phones in range(self.min_phone_seq_length, self.max_phone_seq_length+1):
            window_size_sec = num_phones * self.phone_length_secs
            hop_size_sec = window_size_sec * (1 - self.overlap_fraction)

            window_size_frames = round(window_size_sec * self.hubert_sampling_rate)
            hop_size_frames = round(hop_size_sec * self.hubert_sampling_rate)

            if window_size_frames >= speech_length:
                windows.append(embedded_speech)
                break

            padded_speech_embedding = \
                F.pad(embedded_speech, (0,0, 0, window_size_frames), 'constant', 0)
            
            for i in range(0, speech_length, hop_size_frames):
                windows.append(padded_speech_embedding[i:i+window_size_frames])
        
        return windows
    
    def split_embedded_data_into_list(self, embedded_speech, _=None):
        """Wrapper to match implementation in PhoneSplitter class

        Args:
            embedded_speech (tensor): tensor with embedded speech - shape [num_hubert_frames, 768]
            _ (None): unused argument to match PhoneSplitter implementation

        Returns:
            list[tensor]: list of windowed tensors
        """
        return self.split_embedded_data_into_windows_list(embedded_speech)


if __name__ == "__main__":
    # import moved here to avoid circular imports
    from awe_model.train_model import load_model

    args = sys.argv

    if len(args)>4:
        language = args[4]
        phone_timings_fname = args[5]
        min_phone_seq_length = int(args[6])
        max_phone_seq_length = int(args[7])
        use_window_splitter_instead_of_phone_splitter = parse_boolean_input(args[8])
        layer = int(args[9])
        save_embedding_folder = args[10]
        model_save_dir = args[11]
        model_name = args[12]
    else:
        language = "banjara"
        phone_timings_fname = "phone_all_mpr.ctm"
        min_phone_seq_length = 5
        max_phone_seq_length = 14
        use_window_splitter_instead_of_phone_splitter = True
        layer = 9
        save_embedding_folder = f"{min_phone_seq_length}_{max_phone_seq_length}_again"
        model_save_dir = \
            f"data/tamil/models/awe/{layer}/lr_1e-4_tmp_0.07_acc_1000_bs_5_3_9"
        model_name = "2024-07-20_23:47:58_checkpoint_epoch_0.pt"

    scratch_prefix = f"/scratch/space1/tc062/sanjayb"
    top_level_embedding_dir = f"data/{language}/embeddings"
    phone_timings_file = f"data/{language}/analysis/{phone_timings_fname}"
    # folder = f"documents"
    folder = str(sys.argv[1])  # documents or queries
    skip_existing_files = False
    batched_inference = True

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # n_parts = 1
    n_parts = int(sys.argv[2])
    if n_parts > 1:
        part = int(sys.argv[3])
    else:
        part = 0

    # set to none if queries are not cut out of longer documents
    if folder == "queries" and language == "tamil":
        query_times_file = f"data/{language}/analysis/queries_times.txt"
        print(f"Using query times file: {query_times_file}")
    else:
        query_times_file = None

    if use_window_splitter_instead_of_phone_splitter:
        splitter = WindowSplitter(min_phone_seq_length=min_phone_seq_length, 
                                    max_phone_seq_length=max_phone_seq_length)
    else:
        splitter = PhoneSplitter(phone_timings_file, query_times_file, 
                            min_phone_seq_length, max_phone_seq_length)

    embedding_dir = f"{top_level_embedding_dir}/{folder}/{layer}/raw"
    save_embedding_dir = \
        f"{scratch_prefix}/{top_level_embedding_dir}/{folder}/{layer}/{save_embedding_folder}"


    model_files = [file for file in os.listdir(model_save_dir) if file.endswith(".pt")]
    if len(model_files) == 1:
        model_path = f"{model_save_dir}/{model_files[0]}"
    else:
        model_path = f"{model_save_dir}/{model_name}"

    model = SSEmodel(device=device)
    model.to(device)
    model.eval()
    state_dict = load_model(model_path, model, device)
    model_output_size = 512

    print(f"Extracting AWE embeddings for {folder} in {embedding_dir}")
    print(f"Saving embeddings to {save_embedding_dir}")
    print((f"For layer {layer} using model {model_path}, batched_inference: {batched_inference}\n"
           f"use window splitter: {use_window_splitter_instead_of_phone_splitter}"))

    make_dir(save_embedding_dir)

    with torch.no_grad():
        t1 = time.perf_counter()

        files = os.listdir(embedding_dir)
        file_part = split_list_into_n_parts_and_get_part(files, n_parts, part)

        saved_files = os.listdir(save_embedding_dir)

        dataset_length = len(file_part)
        for idx, file in enumerate(file_part):
            if "all_embeddings" in file: continue

            if skip_existing_files:
                if file in saved_files:
                    print(f"Skipping {file} as it already exists")
                    continue

            with open(f"{embedding_dir}/{file}", "rb") as f:
                hubert_embeddings = pkl.load(f)
            
            hubert_embeddings = splitter.split_embedded_data_into_list(hubert_embeddings, file)
            
            if hubert_embeddings == []:
                print(f"Skipping {file} as no phone sequences found")
                continue
            
            if batched_inference:
                batched_tensors = pad_sequence(hubert_embeddings, batch_first=True, padding_value=0)
                model_inputs = batched_tensors.to(device)
                model_outputs = model(model_inputs)
                del model_inputs
                model_outputs_cpu = model_outputs.to("cpu")
                del model_outputs
            
            else:
                if model_output_size is None:
                    raise ValueError("model_output_size must be provided when batched_inference is True")   

                model_outputs_cpu = torch.zeros((len(hubert_embeddings), model_output_size))
                for i, hubert_embedding in enumerate(hubert_embeddings):
                    hubert_embedding = hubert_embedding.unsqueeze(0)
                    hubert_embedding = hubert_embedding.to(device)
                    model_embedding = model(hubert_embedding)
                    del hubert_embedding
                    model_embedding_cpu = model_embedding.to("cpu")
                    del model_embedding
                    model_outputs_cpu[i] = model_embedding_cpu
            

            percentage = (idx + 1)/dataset_length * 100
            print(f"{percentage:.2f}% done")
            t = time.perf_counter()
            print(f"Time: {t - t1:.2f} s")
            with open(f"{save_embedding_dir}/{file}", "wb") as f:
                pkl.dump(model_outputs_cpu, f)


