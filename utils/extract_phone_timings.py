"""Extract phone timings from audio files using an XLSR-based multilingual phone recogniser (MPR),
from the voice_search_server project, https://github.com/unmute-tech/voice-search-server."""
import os
import sys
import time

from utils.common_functions import split_list_into_n_parts_and_get_part
from kaldi.lat.functions import compact_lattice_to_word_alignment
from voice_search_server.enroll import pcm
from voice_search_server.lib.asr import create_asr

def load_phone_symbol_table(phone_symbol_table_path):
    """Given a path to a phone symbol table, returns a 
    dictionary mapping phone ids to phone symbols.

    Args:
        phone_symbol_table_path (str): path to phone symbol table.

    Returns:
        dict{int \: str}: dictionary mapping phone ids to phone symbols.
    """
    phone_id_to_symbol = {}
    with open(phone_symbol_table_path, 'r') as f:
        for line in f:
            symbol, phone_id = line.strip().split(" ")
            phone_id_to_symbol[int(phone_id)] = symbol
    return phone_id_to_symbol

def convert_phone_ids_to_symbols(phone_ids, phone_symbol_table):
    """Convert list of phone ids to list of phone symbols. Given a
    phone symbol table.

    Args:
        phone_ids (list[int]): list of phone ids.
        phone_symbol_table (dict{int \: str}): dictionary maping phone ids to phone symbols.

    Returns:
        list[str]: list of phone symbols.
    """
    return [phone_symbol_table[phone_id] for phone_id in phone_ids]


if __name__ == "__main__":
    top_level_dir = "data/tamil/"
    audio_dir = f"{top_level_dir}/all_data"
    phone_timings_output_file = f"{top_level_dir}/analysis/phone_all_mpr.ctm"

    # for parallel processing
    n_parts = 64
    if n_parts > 1:
        part = sys.argv[1]
        phone_timings_output_file = f"{top_level_dir}/analysis/phone_timings/phone_part_{part}_mpr.ctm"
    else:
        part = 0

    xlsr_to_sec_conversion_factor = 0.02
    silence_phones = ["sil", "spn", "sp"]

    model_path = 'voice_search_server/model'
    phone_symbol_table_path = f"voice_search_server/model/words.txt"
    return_full_dictionary = True
    model = create_asr(model_path, return_full_dictionary)

    phone_symbol_table = load_phone_symbol_table(phone_symbol_table_path)

    files = sorted(os.listdir(audio_dir))

    file_part = split_list_into_n_parts_and_get_part(files, n_parts, part)

    length = len(file_part)
    t1 = time.perf_counter()

    fhandle = open(phone_timings_output_file, 'w')
    for fnum, file in enumerate(file_part):
        if not file.endswith(".wav"):
            continue
        print(file)
        audio_path = f"{audio_dir}/{file}"
        audio_bytes = pcm(audio_path)
        res = model.transcribe(audio_bytes)
        # phones = res["text"]
        best_path = res["best_path"]

        alignment = compact_lattice_to_word_alignment(best_path)

        phone_ids = alignment[0]
        start_times_xlsr_frames = alignment[1]
        duration_xlsr_frames = alignment[2]

        file = file.replace(".wav", "")

        phones_list = convert_phone_ids_to_symbols(phone_ids, phone_symbol_table)
        if all(phone in silence_phones for phone in phones_list) or len(phones_list) == 0:
            print(f"Skipping {file} because it only has silence phones")
            print(phones_list)
            continue
        for i, phone in enumerate(phones_list):
            start_time = start_times_xlsr_frames[i] * xlsr_to_sec_conversion_factor
            duration = duration_xlsr_frames[i] * xlsr_to_sec_conversion_factor
            fhandle.write(f"{file} 1 {start_time:.3f} {duration:.3f} {phone}\n")
        

        percentage = (fnum+1)/length * 100
        print(f"{percentage:.2f}% done")
        print(f"Time: {time.perf_counter() - t1:.2f} s")
    
    fhandle.close()