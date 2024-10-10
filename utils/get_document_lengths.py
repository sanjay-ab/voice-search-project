"""Get the lengths of all the recordings in a specified directory and save them to a 
file in ascending order, with the format: <filename>: <length in seconds>"""
import os

from utils.common_functions import get_wav_file_length

if __name__ == "__main__":
    language = "gujarati"
    documents_directory = f"data/{language}/documents"
    # queries_directory = "data/{language}/queries"
    save_file = f"data/{language}/analysis/document_lengths.txt"

    min_duration = 0
    max_duration = 1e10
    expected_sr = 16000
    docs_dict = {}

    for directory in [documents_directory]:
        lengths = []
        print(f"Directory: {directory}")
        files = sorted(os.listdir(directory))
        for i, file in enumerate(files):
            if file.endswith(".wav"):
                length = get_wav_file_length(f"{directory}/{file}")
                if length < min_duration or length > max_duration:
                    print(f"File {file} has length {length} seconds, skipping")
                    continue
                docs_dict[file] = length
            print(f"Processed {(i+1)*100/len(files):.2f}% files", end="\r")

    docs_dict = sorted(docs_dict.items(), key=lambda x: x[1])

    with open(save_file, "w") as f:
        for doc, length in docs_dict:
            f.write(f"{doc}: {length}\n")

