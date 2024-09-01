import time
import os
import sys
import pickle as pkl
import torch
from torch.nn.utils.rnn import pad_sequence
from utils.common_functions import make_dir, split_list_into_n_parts_and_get_part, parse_boolean_input

if __name__ == "__main__":
    # import moved here to avoid circular imports
    from awe_model.train_model import load_model

    args = sys.argv

    if len(args)>1:
        folder = str(sys.argv[1])  # documents or queries
        n_parts = int(sys.argv[2])
        if n_parts > 1:
            part = int(sys.argv[3])
        else:
            part = 0
    else:
        folder = f"documents"
        n_parts = 1
        part = 0

    if len(args)>4:
        language = args[4]
        layer = int(args[5])
        save_embedding_folder = args[6]
        model_save_dir = args[7]
        model_name = args[8]
        output_dim = int(args[9])
        model_type = args[10]
    else:
        language = "banjara"
        layer = 9
        save_embedding_folder = f"5_14_test_rec"
        model_save_dir = \
            f"data/tamil/models/rec/9/mid_2048_dropout_layernorm_lr_0.0005_linear_weight_decay_0.01"
        model_name = "2024-07-27_14:25:30_checkpoint_epoch_12.pt"
        output_dim = 512
        model_type = "standard"  # "extra_linear" or "standard"

    if model_type == "standard":
        from rec_model.model import SSEmodel
    elif model_type == "extra_linear":
        from rec_model.extra_linear_layer_model import SSEmodel
    else:
        raise ValueError(f"Model type {model_type} not recognized.")

    scratch_prefix = f"/scratch/space1/tc062/sanjayb"
    top_level_embedding_dir = f"data/{language}/embeddings"
    skip_existing_files = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if language == "tamil":
        embedding_dir = f"{top_level_embedding_dir}/{folder}/{layer}/raw_valid"
    elif language == "banjara":
        embedding_dir = f"{top_level_embedding_dir}/{folder}/{layer}/raw"

    save_embedding_dir = \
        f"{scratch_prefix}/{top_level_embedding_dir}/{folder}/{layer}/{save_embedding_folder}"

    model_files = [file for file in os.listdir(model_save_dir) if file.endswith(".pt")]
    if len(model_files) == 1:
        model_path = f"{model_save_dir}/{model_files[0]}"
    else:
        model_path = f"{model_save_dir}/{model_name}"

    model = SSEmodel(device=device, output_dim=output_dim)
    model.to(device)
    model.eval()
    state_dict = load_model(model_path, model, device)

    print(f"Extracting recording embeddings for {folder} in {embedding_dir}")
    print(f"Saving embeddings to {save_embedding_dir}")
    print((f"For layer {layer} using model {model_path}\n"))

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
            
            if hubert_embeddings == []:
                print(f"Skipping {file} as no phone sequences found")
                continue
            
            hubert_embeddings = hubert_embeddings.unsqueeze(0)
            hubert_embeddings = hubert_embeddings.to(device)
            model_embedding = model(hubert_embeddings)
            del hubert_embeddings
            model_outputs_cpu = model_embedding.to("cpu")
            del model_embedding
            model_outputs_cpu
            

            percentage = (idx + 1)/dataset_length * 100
            print(f"{percentage:.2f}% done")
            t = time.perf_counter()
            print(f"Time: {t - t1:.2f} s")
            with open(f"{save_embedding_dir}/{file}", "wb") as f:
                pkl.dump(model_outputs_cpu, f)


