"""Dataset used for training the learned pooling model. Formats data into batches, where 
each batch is made up of a group of positive pairs, that can be used with an
NTXent contrastive loss."""
import random
import time
import copy
import itertools
import pickle as pkl
from collections import defaultdict

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from awe_model.extract_training_data_phones_from_embedded_data import get_embedded_phones_dict

class PhonePairsDataset(Dataset):
    """Dataset class for training the learned pooling model. Formats data into batches, where
    each batch is made up of a group of positive pairs, that can be used with an NTXent contrastive loss.
    Each pair consists of two different embeddings, which are from the same phone sequence. Other pairs
    in the same batch are from different phone sequences.
    """

    def __init__(self, language, embedding_dir_or_file, num_pairs_per_batch, phone_timings_file, time_limit=240,
                  min_phone_seq_length=3, max_phone_seq_length=9,
                  perturb_sequences=False, max_one_sided_perturb_amount=0.1):
        """Initialise PhonePairsDataset class.

        Args:
            language (str): the language of the dataset.
            embedding_dir_or_file (str): path of directory or file containing the embedded data.
                If using a file, when unpickled it must be a dictionary that uses phone sequences
                as keys and a list of embeddings as values.
            num_pairs_per_batch (int): the max number of pairs to be created for each batch.
            phone_timings_file (str): path to the .ctm phone timings file.
            time_limit (int, optional): the time limit (in seconds) for creating the dataset.
                Defaults to 240.
            min_phone_seq_length (int, optional): minimum phone sequence length (in phones). 
                Defaults to 3.
            max_phone_seq_length (int, optional): maximum phone sequence length (in phones). 
                Defaults to 9.
            perturb_sequences (bool, optional): define whether to perturb the boundaries of sequences.
                If true, the edges of extracted sequences are randomly moved, up to some specified
                amount. Defaults to False.
            max_one_sided_perturb_amount (float, optional): Maximum amount to perturb sequences to.
                Defined as a percentage of the length of the whole phone sequence. I.e., 0.1 means edges
                of the phone sequences can be shifted by up to 10% of the length of the whole phone 
                sequence. Defaults to 0.1.

        Raises:
            ValueError: Perturbing sequences is not supported when also loading from a file.
        """
        self.perturb_sequences = perturb_sequences
        self.max_one_sided_perturb_amount = max_one_sided_perturb_amount
        self.min_phone_seq_length = min_phone_seq_length
        self.max_phone_seq_length = max_phone_seq_length
        self.phone_timings_file = phone_timings_file
        self.num_pairs_per_batch = num_pairs_per_batch
        self.time_limit = time_limit

        t1 = time.perf_counter()
        if embedding_dir_or_file.endswith(".pkl"):
            print(f"Loading embedded data from file: {embedding_dir_or_file}")
            if perturb_sequences:
                raise ValueError("Perturbing sequences is not supported when loading from a file.")
            embedded_data_dict = pkl.load(open(embedding_dir_or_file, "rb"))
        else:
            print(f"Loading embedded data from directory: {embedding_dir_or_file}")
            self.embedding_dir = embedding_dir_or_file
            embedded_data_dict = get_embedded_phones_dict(phone_timings_file, language, embedding_dir_or_file, 
                                                        min_phone_seq_length, max_phone_seq_length,
                                                        perturb_sequences, max_one_sided_perturb_amount)

        self.embedded_data_dict = {key.replace(".pkl", ""): value 
                                   for key, value in embedded_data_dict.items()}

        print(f"Loaded embedded data from {embedding_dir_or_file}")
        print(f"Time taken: {time.perf_counter() - t1:.2f} s")

        self._create_indices_dict()
        self._create_paired_data()

        print(f"Created paired data")

    def __len__(self):
        return len(self.paired_data)

    def __getitem__(self, idx):
        phone_seq_labels = self.paired_data_labels[idx]
        paired_data_list = self.paired_data[idx]

        return phone_seq_labels, paired_data_list 
    
    def _create_indices_dict(self):
        self.indices_dict = defaultdict(list)
        for phone_seq, embeddings_list in self.embedded_data_dict.items():
            for i, j in itertools.permutations(range(len(embeddings_list)), 2):
                self.indices_dict[phone_seq].append((i, j))
    
    def _create_paired_data(self):
        self.paired_data = []
        self.paired_data_labels = []
        indices_dict = copy.deepcopy(self.indices_dict)
        start_time = time.perf_counter()
        while indices_dict:
            num_classes = len(indices_dict.keys())

            if (time.perf_counter() - start_time) > self.time_limit:
                print(f"Dataset generation time limit reached: {self.time_limit} s.")
                print(f"Number of classes remaining: {num_classes}.")
                break
            
            if num_classes >= self.num_pairs_per_batch:
                num_pairs = self.num_pairs_per_batch
            elif num_classes == 1:
                break
            else: 
                num_pairs = num_classes

            classes_to_add = random.sample(range(num_classes), num_pairs)

            paired_data_to_add = []
            paired_data_labels_to_add = []
            keys_to_remove = []
            for i, (phone_seq, pairs_list) in enumerate(indices_dict.items()):
                if i in classes_to_add:
                    pair_idx_of_idx_to_add = random.sample(range(len(pairs_list)), 1)[0]
                    idx_pair_to_add = indices_dict[phone_seq].pop(pair_idx_of_idx_to_add)
                    pair_to_add = (self.embedded_data_dict[phone_seq][idx_pair_to_add[0]][0],
                                   self.embedded_data_dict[phone_seq][idx_pair_to_add[1]][0])

                    paired_data_labels_to_add.append(phone_seq)
                    paired_data_to_add.extend(pair_to_add)

                    if not indices_dict[phone_seq]:
                        keys_to_remove.append(phone_seq)
            
            for key in keys_to_remove:
                indices_dict.pop(key)

            self.paired_data.append(paired_data_to_add)
            self.paired_data_labels.append(paired_data_labels_to_add)

    def regenerate_paired_data(self):
        """Regenerate paired data, to change which pairs are used in each batch."""

        if self.perturb_sequences:
            t1 = time.perf_counter()
            print(f"Regenerating perturbed paired data...")
            embedded_data_dict = \
                get_embedded_phones_dict(self.phone_timings_file, self.embedding_dir, 
                                            self.min_phone_seq_length, self.max_phone_seq_length,
                                            self.perturb_sequences, self.max_one_sided_perturb_amount)

            self.embedded_data_dict = {key.replace(".pkl", ""): value 
                                    for key, value in embedded_data_dict.items()}

            print(f"Reloaded embedded data from {self.embedding_dir}")
            print(f"Time taken: {time.perf_counter() - t1:.2f} s")

            self._create_indices_dict()

        self._create_paired_data()
        print(f"Regenerated paired data")

def collate_as_tensor_and_pad(batch):
    if len(batch) != 1:
        raise ValueError((f"Batch size of dataloader is {len(batch)}, expected 1. "
        f"If you want to change the batch size, please change it during creation of the dataset."))
    phone_seq = batch[0][0]
    tensors = batch[0][1]
    batched_tensors = pad_sequence(tensors, batch_first=True, padding_value=0)
    batch = (phone_seq, batched_tensors)
    return batch

if __name__ == "__main__":
    dataset = PhonePairsDataset("data/tamil/embeddings/training_data/8/raw/all_embeddings.pkl", 2,
                                test_mode=False)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_as_tensor_and_pad, shuffle=False)
    length = len(dataloader)
    print(length)
    for i, batch in enumerate(dataloader):
        print(batch)
        for phone_pairs, hubert_embeddings in batch:
            percentage = (i+1)/length * 100
            print(f"{percentage:.2f}% done")
            print(phone_pairs)
            print(hubert_embeddings)
        # print(batch)