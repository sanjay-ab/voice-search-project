import pickle as pkl
import os

if __name__ == "__main__":
    top_embedding_dir = "data/tamil/embeddings"
    layer = "8"
    embedding_dir = f"{top_embedding_dir}/training_data/{layer}/raw"

    files = os.listdir(embedding_dir)

    phone_embedding_dict = {}

    for file in files:
        if "all_embeddings" in file: continue
        phone_embedding_dict[file] = pkl.load(open(
            f"{embedding_dir}/{file}", "rb"))
    
    pkl.dump(phone_embedding_dict, open(f"{embedding_dir}/all_embeddings.pkl", "wb"))