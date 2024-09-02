"""Learned pooling model with an additional output linear projection"""
from torch import nn

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
        transformer_dim=512,
        middle_dim=512,
        output_dim=512,
        device='cuda:0',
        no_grad_on_awe_model = True,
        awe_model_path=\
            "data/tamil/models/awe/9/lr_1e-4_tmp_0.07_acc_1000_bs_5_3_9/2024-07-20_23:47:58_checkpoint_epoch_0.pt"
    ):

        super(SSEmodel, self).__init__()
        up_proj_dim = 512
        print(f"up_proj_dim: {up_proj_dim}")
        print(f"output_dim: {output_dim}")
        print(f"middle_dim: {middle_dim}")

        self.awe_model = awe_model.SSEmodel(device=device)
        load_model(awe_model_path, self.awe_model, device=device)

        if no_grad_on_awe_model:
            for param in self.awe_model.parameters():
                param.requires_grad = False

        self.linear_layer1 = nn.Linear(middle_dim, output_dim)

        # note that linear_layer2, relu and project_head are not used.
        # They are here for backward compatibility with previous iteration of the model. 
        self.linear_layer2 = nn.Linear(middle_dim, transformer_dim)
        self.relu = nn.ReLU()
        self.project_head=nn.ModuleList([nn.Linear(transformer_dim,up_proj_dim),nn.ReLU(),nn.Linear(up_proj_dim,output_dim)])

        self.init_weight()

    def init_weight(self):
        init_layer(self.linear_layer1)
        init_layer(self.linear_layer2)

        for layer in self.project_head:
            init_layer(layer)

        
    def forward(self, x):
        """
        Input: (batch_size, time_steps, mel_bins)
        """
        x = self.awe_model(x)
        x = self.linear_layer1(x)
        # PROJECTION HEAD

        # for layer in self.project_head:
        #     x=layer(x)
        return x

