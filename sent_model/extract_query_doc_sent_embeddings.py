import time
import os
import sys
import pickle as pkl
import torch
from torch.nn.utils.rnn import pad_sequence
from sent_model.hubert_model import SSEmodel
from utils.common_functions import make_dir, split_list_into_n_parts_and_get_part, parse_boolean_input

if __name__ == "__main__":
    # import moved here to avoid circular imports
    from awe_model.train_model import load_model

    args = sys.argv

    if len(args)>4:
        language = args[4]
        layer = int(args[5])
        save_embedding_folder = args[6]
        model_save_dir = args[7]
        model_name = args[8]
        use_awes = parse_boolean_input(args[9])
    else:
        language = "tamil"
        layer = 9
        save_embedding_folder = f"test_sent"
        model_save_dir = \
            f"data/tamil/models/sent/{layer}/test_sent_model"
        model_name = "2024-07-22_16:39:45_checkpoint_epoch_10.pt"
        use_awes = False

    scratch_prefix = f"/scratch/space1/tc062/sanjayb"
    top_level_embedding_dir = f"data/{language}/embeddings"
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

    if use_awes:
        embedding_dir = f"{scratch_prefix}/{top_level_embedding_dir}/{folder}/{layer}/3_9"
    else:
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

    print(f"Extracting sentence embeddings for {folder} in {embedding_dir}")
    print(f"Saving embeddings to {save_embedding_dir}")
    print((f"For layer {layer} using model {model_path}, batched_inference: {batched_inference}\n"))

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
            
            # if batched_inference:
            #     batched_tensors = pad_sequence(hubert_embeddings, batch_first=True, padding_value=0)
            #     model_inputs = batched_tensors.to(device)
            #     model_outputs = model(model_inputs)
            #     del model_inputs
            #     model_outputs_cpu = model_outputs.to("cpu")
            #     del model_outputs
            
            # else:

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


