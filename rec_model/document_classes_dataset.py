"""Dataset used for training the learned pooling model on the recording embedding task.
Formats data into batches, where each batch is made up of a group of positive pairs,
that can be used with an NTXent contrastive loss."""
import random
import time
import os
import copy
import itertools
import pickle as pkl
from collections import defaultdict

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils.split_data_test_train import extract_gold_labels_for_queries_tamil

class DocumentClassesDataset(Dataset):
    """Dataset class for training the learned pooling model on the recording embedding task.
    Formats data into batches, where each batch is made up of a group of positive pairs, that
    can be used with an NTXent contrastive loss. Each pair consists of two different embeddings,
    which are from two different recordings, containing the same keyword. Other pairs in the
    same batch are from different keyword groups.
    """
    def __init__(self, document_embedding_dir, queries_embedding_dir, 
                 num_pairs_per_batch, reference_file, time_limit=240):
        """Initialise the DocumentClassesDataset class.

        Args:
            document_embedding_dir (str): path to the directory containing the document embeddings.
            queries_embedding_dir (str): path to the directory containing the query embeddings.
            num_pairs_per_batch (int): the max number of pairs to be created for each batch.
            reference_file (str): file describing which queries are related to which documents, 
                along with and their keyword labels.
            time_limit (int, optional): time limit (in seconds) for creating the dataset. Defaults to 240.

        Raises:
            NotImplementedError: Class not implemented for Banjara dataset.
        """
        
        self.num_pairs_per_batch = num_pairs_per_batch
        self.time_limit = time_limit

        if "tamil" in document_embedding_dir:
            self.language = "tamil"
            self.documents_for_queries_dict, self.queries_to_labels_dict = \
                extract_gold_labels_for_queries_tamil(reference_file)
        elif "banjara" in document_embedding_dir:
            raise NotImplementedError("Banjara dataset not implemented yet.")

        self._load_embedded_data(document_embedding_dir, queries_embedding_dir)

        self._create_indices_dict()
        self._create_paired_data()

        print(f"Created paired data")

    def __len__(self):
        return len(self.paired_data)

    def __getitem__(self, idx):
        phone_seq_labels = self.paired_data_labels[idx]
        paired_data_list = self.paired_data[idx]

        return phone_seq_labels, paired_data_list 
    
    def _load_embedded_data(self, document_embedding_dir, queries_embedding_dir):
        self.embedded_data_dict = defaultdict(list)
        # total_length = 0
        # for query, document_list in self.documents_for_queries_dict.items():
        #     total_length += len(document_list)
        # print(f"Total num documents: {total_length}")
        # print(f"Total num queries: {len(self.documents_for_queries_dict.keys())}")
        # print(f"Total num queries + documents: {total_length + len(self.documents_for_queries_dict.keys())}")

        for query, document_list in self.documents_for_queries_dict.items():
            label = self.queries_to_labels_dict[query]
            if self.language == "tamil":
                query_fname = f"q_{query}.pkl"
            else:
                query_fname = f"{query}.pkl"

            if query_fname in os.listdir(queries_embedding_dir):
                self.embedded_data_dict[label].append(
                    pkl.load(open(f"{queries_embedding_dir}/{query_fname}", "rb")))
            else:
                print(f"WARNING: Query {query_fname} not found in {queries_embedding_dir}")
            for document in document_list:
                document_fname = f"{document}.pkl"
                if document_fname in os.listdir(document_embedding_dir):
                    self.embedded_data_dict[label].append(
                        pkl.load(open(f"{document_embedding_dir}/{document_fname}", "rb")))
                else:
                    print(f"WARNING: Document {document_fname} not found in {document_embedding_dir}")
    
    def _create_indices_dict(self):
        self.indices_dict = defaultdict(list)
        for label, embeddings_list in self.embedded_data_dict.items():
            for i, j in itertools.permutations(range(len(embeddings_list)), 2):
                self.indices_dict[label].append((i, j))
    
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
            for i, (label, pairs_list) in enumerate(indices_dict.items()):
                if i in classes_to_add:
                    pair_idx_of_idx_to_add = random.sample(range(len(pairs_list)), 1)[0]
                    idx_pair_to_add = indices_dict[label].pop(pair_idx_of_idx_to_add)
                    pair_to_add = (self.embedded_data_dict[label][idx_pair_to_add[0]],
                                self.embedded_data_dict[label][idx_pair_to_add[1]])

                    paired_data_labels_to_add.append(label)
                    paired_data_to_add.extend(pair_to_add)

                    if not indices_dict[label]:
                        keys_to_remove.append(label)
            
            for key in keys_to_remove:
                indices_dict.pop(key)

            self.paired_data.append(paired_data_to_add)
            self.paired_data_labels.append(paired_data_labels_to_add)

    def regenerate_paired_data(self):
        """Regenerate paired data, to change which pairs are used in each batch."""
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
    document_embedding_dir = "data/tamil/embeddings/documents/9/raw"
    queries_embedding_dir = "data/tamil/embeddings/queries/9/raw"
    analysis_dir = "data/tamil/analysis"
    reference_file = f"{analysis_dir}/ref_of_queries_in_docs.txt"
    dataset = DocumentClassesDataset(document_embedding_dir, queries_embedding_dir, 2, reference_file)
    # dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_as_tensor_and_pad, shuffle=False)