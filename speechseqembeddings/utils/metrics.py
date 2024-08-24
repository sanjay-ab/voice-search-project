import numpy as np
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import sys
import torch

def compute_map(embeddings, labels,faster=True):
    r"""
    Compute the MAP@R as defined in section 3.2 of https://arxiv.org/pdf/2003.08505.pdf

    Parameters:
        - embeddings (2D numpy array): shape = (N,d), contains the embeddings to evaluate
        - labels (1D numpy array): shape =(N,), contains the labels.
                                   Each element should be an integer.
    Returns:
        - mean_average_precision_at_r (float): the value of the MAP@R
    """
    X = np.float32(embeddings)
    X= normalize(X) #embeddings mut be normalize before FAISS
                    # because FAISS only compute the dot 
                    # product and not the cosine distance
    y = np.float32(labels)
    DEVICE = torch.device("cpu")
    
    # Initialize the calculator
    k_faiss=np.bincount(labels.astype(int)).max()
    k_faiss=int(k_faiss)
    #if k_faiss>1000:
    #    k_faiss=1000
    calculator = AccuracyCalculator(include=(), exclude=(), avg_of_avgs=False, k=k_faiss,device=DEVICE)
    # Insure that the type of the numpy array is float32 for faiss
    # Compute the MAP@R
    metric_name='mean_average_precision'
    dict_out = calculator.get_accuracy(
        query=X,
        reference=X,
        query_labels=y,
        reference_labels=y,
        #embeddings_come_from_same_source=True,
        include=[metric_name],
    )
    return dict_out[metric_name]
    
def normalize(data):
    '''Normalize numpy array data in-place.'''
    epsilon=0.00000001
    norm = np.sqrt(np.sum(np.power(data,2), axis=1))
    data /= (norm[:, None]+epsilon)
    return data


