"""Simple file to help move embedded documents to a different directory"""
import os
from utils.examine_datasets import read_documents_file
from utils.common_functions import make_dir

if __name__ == "__main__":
    language = "gujarati"
    layer = 9
    top_level_dir = f"data/{language}/"
    documents_fname = f"{top_level_dir}/analysis/all_documents.txt"
    top_document_embedding_dir = f"{top_level_dir}/embeddings/documents/{layer}"
    raw_document_embedding_dir = f"{top_document_embedding_dir}/raw"
    alt_document_embedding_dir = f"{top_document_embedding_dir}/raw_other"

    make_dir(alt_document_embedding_dir)

    docs = read_documents_file(documents_fname)

    raw_files = os.listdir(raw_document_embedding_dir)

    counter = 0

    for file in raw_files:
        file_basename = file.replace(".pkl", "")
        if file_basename not in docs:
            counter += 1
            print(file_basename)
            # os.rename(f"{raw_document_embedding_dir}/{file}", f"{alt_document_embedding_dir}/{file}")
    
    print(counter)