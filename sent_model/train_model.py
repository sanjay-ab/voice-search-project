from datetime import datetime as dt
import torch
import time
import random
import numpy as np
from torch.utils.data import DataLoader
from awe_model.train_model import train_one_epoch, calculate_validation_loss, save_model, load_model, NTXentLoss
from sent_model.hubert_model import SSEmodel
from sent_model.document_classes_dataset import DocumentClassesDataset, collate_as_list, collate_as_tensor_and_pad
from utils.common_functions import make_dir

if __name__== "__main__":
    seed = 3456542
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    language = "tamil"
    layer = 9
    scratch_prefix = f"/scratch/space1/tc062/sanjayb"
    top_embedding_dir = f"data/{language}/embeddings"
    document_embedding_dir = f"{scratch_prefix}/{top_embedding_dir}/documents/{layer}/raw"
    queries_embedding_dir = f"{scratch_prefix}/{top_embedding_dir}/queries/{layer}/raw"

    train_reference_file = f"data/{language}/analysis/ref_of_queries_in_docs_train.txt"
    valid_reference_file = f"data/{language}/analysis/ref_of_queries_in_docs_valid.txt"

    print(f"Training reference file: {train_reference_file}")
    print(f"Validation reference file: {valid_reference_file}")

    load_model_from_checkpoint = False
    model_load_dir = \
        f"data/{language}/models/sent/{layer}/awe2_1024_proj_lr_0.0005_linear_weight_decay_0.05"
    checkpoint_path = f"{model_load_dir}/2024-07-22_20:15:01_checkpoint_epoch_3.pt"

    device = "cuda"
    temperature = 0.15
    learning_rate = 0.001
    weight_decay = 0.1
    clip_norm = 10
    num_epochs = 100
    patience = 10
    num_pairs_per_batch = 2
    num_batch_pairs_to_accumulate_gradients_over = 200  # set to 1 if you don't want gradient accumulation
    time_limit_to_create_dataset = 600
    batch_as_list = False

    model_save_dir = \
        f"data/{language}/models/sent/{layer}/hubert_mid_2048_lr_{learning_rate}_linear_weight_decay_{weight_decay}"
    datetime_string = dt.now().strftime("%Y-%m-%d_%H:%M:%S")
    model_file_basename = f"{datetime_string}"

    if batch_as_list:
        collate_fn = collate_as_list
    else:
        collate_fn = collate_as_tensor_and_pad

    print(f"START TIME: {datetime_string}")
    print(f"Training model for {language} with inputs from mHuBERT layer {layer}")
    print((f"Number of epochs: {num_epochs}, patience: {patience}, learning rate: {learning_rate}\n"
           f"clip norm: {clip_norm}, temperature: {temperature}, num pairs per batch: {num_pairs_per_batch}\n"
           f"time limit to create dataset: {time_limit_to_create_dataset}\n"
           f"weight decay: {weight_decay}\n"
           f"temperature: {temperature}\n"))

    make_dir(model_save_dir)

    t1 = time.perf_counter()
    train_dataset = DocumentClassesDataset(document_embedding_dir, queries_embedding_dir,
                                            num_pairs_per_batch, train_reference_file,
                                            time_limit_to_create_dataset)
    valid_dataset = DocumentClassesDataset(document_embedding_dir, queries_embedding_dir,
                                            num_pairs_per_batch, valid_reference_file,
                                            time_limit_to_create_dataset)
    # set dataloaders to batch size 1
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    validation_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=True,
                                        collate_fn=collate_fn)
    print(f"Time taken to create datasets: {time.perf_counter() - t1:.2f} s")

    model = SSEmodel(device=device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters in model: {params}")
    model.to(device)
    model_output_size = 512
    optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay)
    if load_model_from_checkpoint:
        state_dict = load_model(checkpoint_path, model, device, optimizer)
    loss_function = NTXentLoss(temperature)

    best_valid_loss = float("inf")
    best_epoch = 0
    num_epochs_with_no_improvement = 0
    for epoch_num in range(num_epochs):
        train_one_epoch(model, train_dataloader, loss_function, optimizer,  
                        device, clip_norm, epoch_num, 
                        num_batch_pairs_to_accumulate_gradients_over=num_batch_pairs_to_accumulate_gradients_over,
                        model_output_size=model_output_size, batch_as_list=batch_as_list)
        valid_loss = calculate_validation_loss(model, validation_dataloader, loss_function, 
                                               device, 
                                               model_output_size=model_output_size, 
                                               batch_as_list=batch_as_list)
        save_model(model, optimizer, epoch_num, model_save_dir, model_file_basename, valid_loss)
        train_dataset.regenerate_paired_data()

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch_num
            num_epochs_with_no_improvement = 0
        else:
            num_epochs_with_no_improvement += 1
            if num_epochs_with_no_improvement >= patience:
                print(f"Validation loss has not improved for {patience} epochs. Stopping training.")
                break
    
    print(f"BEST VALIDATION LOSS: {best_valid_loss} at epoch {best_epoch}\n")
