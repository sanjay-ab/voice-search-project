import math
import re
import pickle as pkl

from awe_model.extract_query_doc_phone_hubert_embeddings import read_query_times_file
from utils.common_functions import make_dir

if __name__ == "__main__":
    language = "tamil"
    layer = 9
    hubert_sampling_rate = 50

    document_embedding_dir = f"data/{language}/embeddings/documents/{layer}/raw"
    query_output_embedding_dir = f"data/{language}/embeddings/queries_cut_after_embedding/{layer}/raw"

    query_times_file = f"data/{language}/analysis/queries_times.txt"
    
    make_dir(query_output_embedding_dir)

    query_times_dict = read_query_times_file(query_times_file, language)
    
    for file, (start_time, end_time) in query_times_dict.items():
        print(f"Processing {file}")

        original_filename = re.sub(r"_\d+$", "", file)
        original_document_data = pkl.load(open(f"{document_embedding_dir}/{original_filename}.pkl", "rb"))

        start_time_frames = math.floor(start_time * hubert_sampling_rate)
        end_time_frames = math.ceil(end_time * hubert_sampling_rate)

        query_embedding = original_document_data[start_time_frames:end_time_frames]

        pkl.dump(query_embedding, open(f"{query_output_embedding_dir}/q_{file}.pkl", "wb"))
        