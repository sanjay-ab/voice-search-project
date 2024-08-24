import os
import soundfile as sf

def make_dir(path):
    if not os.path.exists(f"{path}"):
        try:
            os.makedirs(f"{path}")
        except:
            print(f"Failed to make directory {path}")

def check_if_dir_exists(directory):
    if not os.path.exists(directory):
        raise ValueError((f"Directory: {directory} does not exist"))

def split_list_into_n_parts(lst, n):
    """Splits lst into n parts."""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def split_list_into_n_parts_and_get_part(lst, n_parts, part):
    """Splits lst into n parts."""
    if n_parts == 1:
        return lst
    else:
        lst_parts = split_list_into_n_parts(lst, n_parts)
        return lst_parts[int(part)]

def read_wav_file(fname, expected_sample_rate=16000):
    """Returns speech from wav file as channel x samples, expects single channel data"""
    speech, sample_rate = sf.read(fname)
    if sample_rate != expected_sample_rate:
        raise ValueError(f"WARNING: wav files are sampled at {sample_rate} instead of 16 kHz")
    return speech 

def get_wav_file_length(file_path, expected_sr=16000):
    data, sr = sf.read(file_path)
    if sr != expected_sr:
        raise ValueError(f"Expected sample rate of {expected_sr} Hz, but got {sr} Hz")
    num_samples = data.shape[0]

    return num_samples / sr

def parse_boolean_input(input):
    if input.lower() in ['true', '1', 't', 'y', 'yes']:
        return True
    elif input.lower() in ['false', '0', 'f', 'n', 'no']:
        return False
    else:
        raise ValueError("Invalid boolean value")