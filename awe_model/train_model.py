"""Train learned pooling model contrastively using NTXent loss."""
import time
import random
import sys
from datetime import datetime as dt

import numpy as np
import torch
from torch.utils.data import DataLoader

from awe_model.model import SSEmodel
from awe_model.phone_pairs_dataset import PhonePairsDataset, collate_as_tensor_and_pad
from utils.common_functions import make_dir

class NTXentLoss:
    def __init__(self, temperature, reduction="mean"):
        self.temperature = temperature
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction=reduction)

    def _nt_xent_loss(self, x):
        """Calculates normalised temperature-scaled cross-entropy loss for input x. x must be a tensor
        of shape, [2N, embedding_size], where N pairs of positive samples are in x, and are assumed to be 
        adjacent. I.e., the first two rows are a pair, the next two rows are a pair, etc.

        Args:
            x (tensor): tensor of shape [2N, embedding_size], containing N pairs of positive samples to
                calculate the loss for.

        Returns:
            tensor: a scalar tensor containing the loss value.
        """
        assert len(x.size()) == 2

        # Cosine similarity
        norms = x.norm(dim=1, p=2).unsqueeze(1)
        x = x / norms
        x = torch.nan_to_num(x, nan=0.0)
        xcs = torch.mm(x, x.t())
        xcs[torch.eye(x.shape[0]).bool()] = float("-inf")

        # Ground truth labels
        target = torch.arange(x.shape[0])
        target[0::2] += 1
        target[1::2] -= 1

        # Standard cross-entropy loss
        return self.ce_loss(xcs / self.temperature, target)

    def __call__(self, x):
        return self._nt_xent_loss(x) 

def get_loss_for_one_batch(model, hubert_embeddings, loss_function, device):
    """Get the loss for one batch.

    Args:
        model (SSEmodel): model to get loss for
        hubert_embeddings (tensor): input batch of mHuBERT embeddings
        loss_function (NTXentLoss): loss function to use. 
        device (str): compute device

    Returns:
        tensor: scalar tensor containing the loss value.
    """
    model_inputs = hubert_embeddings.to(device)
    model_outputs = model(model_inputs)
    del model_inputs
    model_outputs_cpu = model_outputs.to("cpu")
    del model_outputs
    
    loss = loss_function(model_outputs_cpu)
    return loss

def train_one_epoch(model, dataloader, loss_function, optimizer, device, clip_norm, epoch_num,
                     num_batch_pairs_to_accumulate_gradients_over = 1,
                     num_pairs_to_calc_loss_with=800):
    """Train model for one epoch

    Args:
        model (SSEmodel): self supervised model to train
        dataloader (Dataloader): pytorch dataloader - must be batch size 1, contains a list of phone pairs 
            in each batch
        loss_function (NTXentLoss): instance of loss function
        optimizer (torch.optim): Pytorch optimiser, e.g. Adam
        device (str): compute device
        clip_norm (float): value to clip gradients to - clips gradients of all parameters together to
            this specified norm, as if all gradients were put into a single vector and normalised.
            Uses clip_grad_norm. 
        epoch_num (int): current epoch number
        num_batch_pairs_to_accumulate_gradients_over (int): number of batch pairs to accumulate gradients over.
            E.g., 1000 means summing gradients over at least 1000 pairs before performing the gradient
            update - if a batch consists of 200 pairs then updates would be done over 5 batches. Defaults to 1,
            where updates are always done after each batch.
        num_pairs_to_calc_loss_with (int): number of phone pairs to calculate loss with. 
            Loss per batch is simply multiplied by a factor so it produces a value comparable 
            to if the batch size was num_pairs_to_calc_loss_with. This is done in an attempt to make it easier to 
            compare losses between two models using different batch sizes. Defaults to 800.
    """
    start_time = time.perf_counter()
    dataloader_length = len(dataloader)
    percent_increment = 10
    next_increment = 0
    total_loss = 0
    total_gradients_clipped = 0
    model.train()
    loss_per_increment = 0
    counter_for_increment_loss = 0
    num_accumulated_batch_pairs = 0

    # assume dataloader has batch size of 1
    for i, (_, hubert_embeddings) in enumerate(dataloader):

        num_pairs_per_batch = hubert_embeddings.shape[0]/2
        num_accumulated_batch_pairs += num_pairs_per_batch

        loss_multiply_factor = num_pairs_to_calc_loss_with / num_pairs_per_batch

        loss = get_loss_for_one_batch(model, hubert_embeddings, loss_function, device)
        loss.backward()

        total_loss += loss.item() * loss_multiply_factor
        loss_per_increment += loss.item() * loss_multiply_factor
        counter_for_increment_loss += 1

        if (num_accumulated_batch_pairs >= num_batch_pairs_to_accumulate_gradients_over) or (i == dataloader_length - 1):
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            if grad_norm > clip_norm:
                total_gradients_clipped += 1
            optimizer.step()
            optimizer.zero_grad()
            num_accumulated_batch_pairs = 0

        percentage = (i+1)/dataloader_length * 100
        if percentage > next_increment:
            print(f"Epoch {epoch_num}: {percentage:.2f}% done")
            print(f"Loss: {loss_per_increment/counter_for_increment_loss}")
            print(f"Time: {time.perf_counter() - start_time:.2f} s")
            next_increment += percent_increment
            loss_per_increment = 0
            counter_for_increment_loss = 0
    
    print(f"\nEpoch {epoch_num} done")
    print(f"Epoch loss: {total_loss/dataloader_length}\n")
    print(f"Time taken for epoch: {time.perf_counter() - start_time:.2f} s")
    print(f"Number of gradients clipped: {total_gradients_clipped}\n")

def calculate_validation_loss(model, dataloader, loss_function, device,
                               num_pairs_to_calc_loss_with=800):
    """Calculate the validation loss for the model.

    Args:
        model (SSEmodel): model to calculate validation loss for
        dataloader (Dataloader): pytorch dataloader - must be batch size 1, contains a list of phone pairs 
            in each batch
        loss_function (NTXentLoss): instance of loss function to use
        device (str): compute device
        num_pairs_to_calc_loss_with (int, optional): number of phone pairs to calculate loss with. 
            Loss per batch is simply multiplied by a factor so it produces a value comparable 
            to if the batch size was num_pairs_to_calc_loss_with. This is done in an attempt to make it easier to 
            compare losses between two models using different batch sizes. Defaults to 800.

    Returns:
        float: average loss over the validation set
    """
    with torch.no_grad():
        start_time = time.perf_counter()
        model.eval()
        total_loss = 0
        dataloader_length = len(dataloader)
        percent_increment = 20
        next_increment = 0

        # assume batch size of 1 for dataloader
        for i, (_, hubert_embeddings) in enumerate(dataloader):
            num_pairs_per_batch = hubert_embeddings.shape[0]/2
            loss_multiply_factor = num_pairs_to_calc_loss_with / num_pairs_per_batch

            loss = get_loss_for_one_batch(model, hubert_embeddings, loss_function, device)
            
            total_loss += loss.item() * loss_multiply_factor

            percentage = (i+1)/dataloader_length * 100
            if percentage > next_increment:
                print(f"Calculating validation loss: {percentage:.2f}% done")
                print(f"Time: {time.perf_counter() - start_time:.2f} s")
                next_increment += percent_increment
        
        average_loss = total_loss/dataloader_length
        print(f"\nValidation loss: {average_loss}\n")
        print(f"Time taken: {time.perf_counter() - start_time:.2f} s")

        return average_loss

def save_model(model, optimizer, epoch, save_dir, model_file_basename, valid_loss):
    """Save model to disk.

    Args:
        model (SSEmodel): model to save.
        optimizer (torch.optim): optimiser used for model training (e.g., Adam).
        epoch (int): last epoch number.
        save_dir (str): path of save directory.
        model_file_basename (str): basename of model file.
        valid_loss (float): validation loss at last epoch.
    """

    save_fname = f"{save_dir}/{model_file_basename}_checkpoint_epoch_{epoch}.pt"
    print(f"Saving model to {save_fname}\n")
    state_dict = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'validation_loss': valid_loss,
    }
    torch.save(state_dict, save_fname)

def load_model(checkpoint_path, model, device, optimizer=None):
    """Load model from checkpoint.

    Args:
        checkpoint_path (str): path of checkpoint file.
        model (SSEmodel): model variable to load checkpoint into.
        device (str): compute device.
        optimizer (torch.optim, optional): pytorch optimiser used with model.
            Defaults to None.

    Returns:
        dict: state dictionary of model.
    """
    print(f"\nLoading model from {checkpoint_path}\n")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict['model'])
    if optimizer is not None:
        optimizer.load_state_dict(state_dict['optimizer'])
    return state_dict

if __name__== "__main__":
    # Set seed for reproducibility
    seed = 3456542
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    args = sys.argv

    if len(args) > 1:
        language = args[1]
        phone_timings_file_name = args[2]
    else:
        language = "tamil"
        phone_timings_file_name = "phone_all.ctm"

    phone_timings_file = f"data/{language}/analysis/{phone_timings_file_name}"
    top_embedding_dir = f"data/{language}/embeddings"
    layer = 9
    min_phone_seq_length = 3
    max_phone_seq_length = 9

    if "mpr" in phone_timings_file_name:
        mpr = "_mpr_"
    else:
        mpr = "_"

    training_embedding_dir = (f"{top_embedding_dir}/training_data/{layer}/raw")
    # training_all_embeddings_file = f"{training_embedding_dir}/all_embeddings_phonetized.pkl"

    validation_embedding_dir = (f"{top_embedding_dir}/validation_data/"
                              f"{layer}/phonetized{mpr}{min_phone_seq_length}_{max_phone_seq_length}")
    validation_all_embeddings_file = f"{validation_embedding_dir}/all_embeddings_phonetized.pkl"

    print(f"Training embeddings directory: {training_embedding_dir}")
    print(f"Validation embeddings file: {validation_all_embeddings_file}")

    load_model_from_checkpoint = False
    model_load_dir = \
        f"data/{language}/models/awe/{layer}/half_lr_1e-4_tmp_0.07_acc_1000_bs_5_{min_phone_seq_length}_{max_phone_seq_length}"
    checkpoint_path = f"{model_load_dir}/2024-07-21_12:25:45_checkpoint_epoch_2.pt"

    device = "cuda"
    perturb_sequences = False
    max_one_sided_perturb_amount = 0.1
    temperature = 0.07
    learning_rate = 0.0001
    clip_norm = 40
    num_epochs = 5
    patience = 4
    num_pairs_per_batch = 5
    num_batch_pairs_to_accumulate_gradients_over = 1000  # set to 1 if you don't want gradient accumulation
    time_limit_to_create_dataset = 600

    model_save_dir = \
        (f"data/{language}/models/awe/{layer}/{mpr[1:]}lr_{learning_rate}"
         f"_tmp_{temperature}_acc_{num_batch_pairs_to_accumulate_gradients_over}_"
         f"bs_{num_pairs_per_batch}_{min_phone_seq_length}_{max_phone_seq_length}")
    datetime_string = dt.now().strftime("%Y-%m-%d_%H:%M:%S")
    model_file_basename = f"{datetime_string}"

    collate_fn = collate_as_tensor_and_pad

    print(f"START TIME: {datetime_string}")
    print(f"Training model for {language} with inputs from mHuBERT layer {layer}")
    print((f"Number of epochs: {num_epochs}, patience: {patience}, learning rate: {learning_rate}\n"
           f"clip norm: {clip_norm}, temperature: {temperature}, num pairs per batch: {num_pairs_per_batch}\n"
           f"num batch pairs to accumulate gradients over: {num_batch_pairs_to_accumulate_gradients_over}\n"
           f"time limit to create dataset: {time_limit_to_create_dataset}\n"
           f"min phone seq length: {min_phone_seq_length}, max phone seq length: {max_phone_seq_length}\n"
           f"perturb sequences: {perturb_sequences}, max one sided perturb amount: {max_one_sided_perturb_amount}\n"))
    
    print(f"Model save directory: {model_save_dir}\n")

    make_dir(model_save_dir)

    t1 = time.perf_counter()
    train_dataset = PhonePairsDataset(language, training_embedding_dir, num_pairs_per_batch, phone_timings_file,
                                        time_limit_to_create_dataset, min_phone_seq_length,
                                        max_phone_seq_length, perturb_sequences, 
                                        max_one_sided_perturb_amount)
    valid_dataset = PhonePairsDataset(language, validation_all_embeddings_file, num_pairs_per_batch, phone_timings_file,
                                       time_limit_to_create_dataset, min_phone_seq_length,
                                       max_phone_seq_length)
    # set dataloaders to batch size 1
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    validation_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=True,
                                        collate_fn=collate_fn)
    print(f"Time taken to create datasets: {time.perf_counter() - t1:.2f} s")

    model = SSEmodel(device=device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
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
