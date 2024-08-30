import torch
import math
from torch import nn
import torch.nn.functional as F
import awe_model.model as awe_model
from awe_model.train_model import load_model

def init_layer(layer):
    """
    Code from https://github.com/wesbz/audioset_tagging_cnn/blob/master/pytorch/models.py
    Initialize a Linear or Convolutional layer.
    """
    if hasattr(layer, "weight"):
        nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)

class SSEmodel(nn.Module):
    """
    Adaptation of the code from https://github.com/wesbz/audioset_tagging_cnn/blob/master/pytorch/models.py
    """

    def __init__(
        self,
        input_size=512,
        middle_dim=512,
        output_dim=256,
        device='cuda:0',
        awe_model_path=\
            "data/tamil/models/awe/9/lr_1e-4_tmp_0.07_acc_1000_bs_5_3_9/2024-07-20_23:47:58_checkpoint_epoch_0.pt"
    ):

        super(SSEmodel, self).__init__()
        # output_dim = 128
        # middle_dim = 512
        print(f"output_dim: {output_dim}")
        print(f"middle_dim: {middle_dim}")

        self.awe_model = awe_model.SSEmodel(device=device)
        load_model(awe_model_path, self.awe_model, device=device)

        for param in self.awe_model.parameters():
            param.requires_grad = False

        self.linear_layer1 = nn.Linear(input_size, middle_dim)
        self.linear_layer2 = nn.Linear(middle_dim, output_dim)
        self.relu = nn.ReLU()
        self.init_weight()

    def init_weight(self):
        init_layer(self.linear_layer1)
        init_layer(self.linear_layer2)

        
    def forward(self, x):
        """
        Input: (batch_size, time_steps, mel_bins)
        """
        x = self.awe_model(x)
        x = self.linear_layer1(x)
        x = self.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.linear_layer2(x)
        return x

