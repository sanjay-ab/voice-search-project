import soundfile as sf
import os
from utils.common_functions import get_wav_file_length

if __name__ == "__main__":
    documents_directory = "data/tamil/all_data"
    # queries_directory = "data/tamil/queries"
    save_file = "data/tamil/analysis/document_lengths_all.txt"

    min_duration = 0
    max_duration = 1e10
    expected_sr = 16000
    docs_dict = {}

    for directory in [documents_directory]:
        lengths = []
        print(f"Directory: {directory}")
        files = sorted(os.listdir(directory))
        for file in files:
            if file.endswith(".wav"):
                length = get_wav_file_length(f"{directory}/{file}")
                if length < min_duration or length > max_duration:
                    print(f"File {file} has length {length} seconds, skipping")
                    continue
                docs_dict[file] = length

    docs_dict = sorted(docs_dict.items(), key=lambda x: x[1])

    with open(save_file, "w") as f:
        for doc, length in docs_dict:
            f.write(f"{doc}: {length}\n")

