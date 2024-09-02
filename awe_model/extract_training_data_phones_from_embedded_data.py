"""Extract training data from mHuBERT embedded audio files. Training data corresponds to groups of 
mHuBERT embeddings that match specific phone sequences."""
import os
import time
import pickle as pkl
from collections import defaultdict

from awe_model.extract_query_doc_awe_embeddings import PhoneSplitter
from utils.common_functions import make_dir

def get_embedded_phones_dict(phone_timings_file, embedding_dir, min_phone_seq_length, max_phone_seq_length,
                             perturb_sequences=False, max_one_sided_perturb_amount=0.1, quiet=True):
    """Extract groups of mHuBERT embeddings from a directory of embedded audio files, where each group
    matches a specific phone sequence.

    Args:
        phone_timings_file (str): path of .ctm file with phone timing information for all recordings.
        embedding_dir (str): path of directory with the embedded audio files, in .pkl format.
        min_phone_seq_length (int): minimum phone sequence length (in phones) to consider.
        max_phone_seq_length (int): maximum phone sequence length (in phones) to consider.
        perturb_sequences (bool, optional): define whether to perturb the boundaries of sequences.
            If true, the edges of extracted sequences are randomly moved, up to some specified
            amount. Defaults to False.
        max_one_sided_perturb_amount (float, optional): Maximum amount to perturb sequences to.
            Defined as a percentage of the length of the whole phone sequence. I.e., 0.1 means edges
            of the phone sequences can be shifted by up to 10% of the length of the whole phone 
            sequence. Defaults to 0.1.
        quiet (bool, optional): Whether to suppress printing progress to std output. Defaults to True.

    Returns:
        dict{str \: list[tensor]}: defaultdict(list) mapping phone sequences to lists of mHuBERT embeddings.
            Where the key is a string of phone labels, and the value is a list of mHuBERT embeddings
            corresponding to that phone sequence.
    """

    splitter = PhoneSplitter(phone_timings_file, None, min_phone_seq_length, max_phone_seq_length)

    files = os.listdir(embedding_dir)
    dataset_length = len(files)

    embedded_phones_dict = defaultdict(list)

    t1 = time.perf_counter()
    for i, file in enumerate(files):
        if "all_embeddings" in file: continue
        if not quiet:
            print(file)
        hubert_embeddings = pkl.load(open(f"{embedding_dir}/{file}", "rb"))

        splitter.split_embedded_data_into_phones_and_append_to_dict(
            hubert_embeddings, file, embedded_phones_dict, 
            perturb_sequences, max_one_sided_perturb_amount)

        percentage = (i + 1)/dataset_length * 100
        if not quiet:
            print(f"{percentage:.2f}% done")
            t = time.perf_counter()
            print(f"Time: {t - t1:.2f} s")
    
    return embedded_phones_dict


if __name__ == "__main__":
    top_level_dir = "data/tamil/"
    folder = "validation_data"
    top_level_embedding_dir = f"{top_level_dir}/embeddings/{folder}"
    phone_timings_file = f"{top_level_dir}/analysis/phone_all.ctm"
    layer = 9
    min_phone_seq_length = 3
    max_phone_seq_length = 9
    perturb_sequences = False  # define whether to perturb the boundaries of the phone sequences
    max_one_sided_perturb_amount = 0.1

    embedding_dir = f"{top_level_embedding_dir}/{layer}/raw"
    save_embedding_dir = \
        f"{top_level_embedding_dir}/{layer}/phonetized_{min_phone_seq_length}_{max_phone_seq_length}"

    print(f"Extracting phonetized embeddings from {embedding_dir}...")
    print((f"Using: phone timings file: {phone_timings_file}\nMin phone seq length: {min_phone_seq_length}"
           f"\nMax phone seq length: {max_phone_seq_length}"))
    print(f"Saving phonetized embeddings to {save_embedding_dir}")
    
    make_dir(save_embedding_dir)

    embedded_phones_dict = get_embedded_phones_dict(
                            phone_timings_file, embedding_dir, min_phone_seq_length, max_phone_seq_length,
                            perturb_sequences, max_one_sided_perturb_amount, False)
    
    t1 = time.perf_counter()
    pkl.dump(embedded_phones_dict, open(f"{save_embedding_dir}/all_embeddings_phonetized.pkl", "wb"))
    print(f"Time to save embeddings: {time.perf_counter() - t1:.2f} s")

    print(f"Finished extracting phonetized embeddings")