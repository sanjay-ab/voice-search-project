from datetime import datetime as dt
import torch
import time
import random
import sys
import numpy as np
from torch.utils.data import DataLoader
from awe_model.train_model import train_one_epoch, calculate_validation_loss, save_model, load_model, NTXentLoss
from rec_model.document_classes_dataset import DocumentClassesDataset, collate_as_tensor_and_pad
from utils.common_functions import make_dir, parse_boolean_input

if __name__== "__main__":
    seed = 3456542
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    args = sys.argv

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
        f"data/{language}/models/awe/{layer}/lr_1e-4_tmp_0.07_acc_1000_bs_5_3_9"
    checkpoint_path = f"{model_load_dir}/2024-07-20_23:47:58_checkpoint_epoch_0.pt"

    device = "cuda"

    if len(args) > 1:
        temperature = float(args[1])
        learning_rate = float(args[2])
        output_dim = int(args[3])
        no_grad_on_awe_model = parse_boolean_input(args[4])
        model_type = args[5]
    else:
        temperature = 0.07
        learning_rate = 0.001
        output_dim = 512
        no_grad_on_awe_model = True
        model_type = "standard"  # "extra_linear" or "standard"
    
    if model_type == "standard":
        from rec_model.model import SSEmodel
        center_save_string = ""
    elif model_type == "extra_linear":
        from rec_model.extra_linear_layer_model import SSEmodel
        center_save_string = f"1_layer_output_dim_{output_dim}_"
    else:
        raise ValueError(f"Model type {model_type} not recognized.")

    clip_norm = 10
    weight_decay = 0.00
    num_epochs = 500
    patience = 5
    num_pairs_per_batch = 5
    num_batch_pairs_to_accumulate_gradients_over = 200  # set to 1 if you don't want gradient accumulation
    time_limit_to_create_dataset = 600
    awe_lr = 1e-5

    if no_grad_on_awe_model:
        no_grad_str = "no_grad"
    else:
        no_grad_str = "grad"

    model_save_dir = \
        f"data/{language}/models/rec/{layer}/finetune_awe_{no_grad_str}_{center_save_string}lr_{learning_rate}_tmp_{temperature}"
    datetime_string = dt.now().strftime("%Y-%m-%d_%H:%M:%S")
    model_file_basename = f"{datetime_string}"

    collate_fn = collate_as_tensor_and_pad

    print(f"START TIME: {datetime_string}")
    print(f"Training model for {language} with inputs from mHuBERT layer {layer}")
    print((f"Number of epochs: {num_epochs}, patience: {patience}, learning rate: {learning_rate}\n"
           f"clip norm: {clip_norm}, temperature: {temperature}, num pairs per batch: {num_pairs_per_batch}\n"
           f"time limit to create dataset: {time_limit_to_create_dataset}\n"
           f"weight decay: {weight_decay}\n"
           f"awe_lr: {awe_lr}\n"
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

    model = SSEmodel(device=device, output_dim=output_dim, no_grad_on_awe_model=no_grad_on_awe_model)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters in model: {params}")
    model.to(device)

    # Set different learning rates
    if no_grad_on_awe_model:
        optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay)
    else:
        awe_model_params = list(model.awe_model.parameters())
        other_params = [param for name, param in model.named_parameters() if 'awe_model' not in name]
        print(f"Number of parameters in AWE model: {sum(p.numel() for p in awe_model_params)}")
        print(f"Number of parameters in other model: {sum(p.numel() for p in other_params)}")
        optimizer = torch.optim.Adam([
            {'params': awe_model_params, 'lr': awe_lr, "weight_decay": 0},
            {'params': other_params, 'lr': learning_rate, "weight_decay": weight_decay}
        ])


    if load_model_from_checkpoint:
        state_dict = load_model(checkpoint_path, model, device, optimizer)
    loss_function = NTXentLoss(temperature)

    best_valid_loss = float("inf")
    best_epoch = 0
    num_epochs_with_no_improvement = 0
    for epoch_num in range(num_epochs):
        train_one_epoch(model, train_dataloader, loss_function, optimizer,  
                        device, clip_norm, epoch_num, 
                        num_batch_pairs_to_accumulate_gradients_over=num_batch_pairs_to_accumulate_gradients_over)
        valid_loss = calculate_validation_loss(model, validation_dataloader, loss_function, device)
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
