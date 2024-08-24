import torch
import math
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import fairseq
import numpy as np
class PositionalEmbedding(nn.Module):
    """
    Code from https://github.com/wesbz/audioset_tagging_cnn/blob/master/pytorch/models.py
    Create the positional embedding.
    """
    def __init__(self, d_model, max_len=1000,device='cpu'):
        super(PositionalEmbedding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0).transpose(0, 1).to(device)

    def forward(self, x):
        return self.pe[: x.size(0), :]

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


def init_bn(bn):
    """
    Code from https://github.com/wesbz/audioset_tagging_cnn/blob/master/pytorch/models.py
    Initialize a Batchnorm layer.
    """
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


class ConvLayer(nn.Module):
    def __init__(self, input_size, channels, stride=(2, 1),kernel_size=(3,1),padding='same'):
        super(ConvLayer, self).__init__()

        self.layer_norm = nn.LayerNorm((input_size, 1))
        self.conv = nn.Conv2d(
            in_channels=input_size,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.init_weight()

    def init_weight(self):
        init_bn(self.layer_norm)
        init_layer(self.conv)

    def forward(self, x):
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # (batch_size, input_size, time_steps, 1)
        x = self.conv(x)  # (batch_size, *, time_steps, 1)
        x = F.glu(x, dim=1)  # (batch_size, *, time_steps, 1)
        x = F.dropout(x, p=0.3, training=self.training)
        x = x.transpose(1, 2)  # (batch_size, time_steps, *, 1)
        return x

class SSEmodel(nn.Module):
    """
    Adaptation of the code from https://github.com/wesbz/audioset_tagging_cnn/blob/master/pytorch/models.py
    """

    def __init__(
        self,
        input_size=768,
        n_conv_layers=1,
        transformer_dim=512,
        n_heads=4,
        n_transformer_layers=1,
        device='cuda:0',
    ):

        super(SSEmodel, self).__init__()
        # Make sure the nb of channels of the last conv layer is 2 x transformer_dim
        self.n_transformer_layers=n_transformer_layers
        self.n_conv_layers=n_conv_layers
        kernel_size=(4,1)
        list_conv_layers = [ConvLayer(input_size=input_size, channels=2 * transformer_dim,kernel_size=kernel_size,stride=(1,1))]
        self.conv_layers = nn.ModuleList(list_conv_layers)
        self.transformer_layers = nn.ModuleList([nn.TransformerEncoderLayer(transformer_dim, n_heads, n_heads * transformer_dim, 0.3) for _ in range(n_transformer_layers)])
        output_dim=transformer_dim
        self.project_head=nn.ModuleList([nn.Linear(transformer_dim,transformer_dim),nn.ReLU(),nn.Linear(transformer_dim,output_dim)])
        self.pos_emb = PositionalEmbedding(transformer_dim,device=device)
        self.init_weight()

    def init_weight(self):
        for layer in self.transformer_layers:
            nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
            layer.self_attn.in_proj_bias.data.fill_(0)
            nn.init.xavier_uniform_(layer.self_attn.out_proj.weight)
            layer.self_attn.out_proj.bias.data.fill_(0)
            init_layer(layer.linear1)
            init_layer(layer.linear2)
            init_bn(layer.norm1)
            init_bn(layer.norm2)
        for layer in self.project_head:
            init_layer(layer)

        
    def forward(self, x):
        """
        Input: (batch_size, time_steps, mel_bins)
        """
        x = x.unsqueeze(-1)  # (batch_size, time_steps, input_size, 1)
        for layer in self.conv_layers:
            x = layer(x)
        x = x.permute(1, 0, 2, 3)  # (time_steps, batch_size, transformer_dim, 1)
        x = x.squeeze(dim=3)  # (time_steps, batch_size, transformer_dim)
        x+= self.pos_emb(x)
        for layer in self.transformer_layers:
            x = layer(x)
        x=x.transpose(0,1)
        
        # TIME POOLING
        x = torch.max(x, dim=1).values # (batch_size, transformer_dim)
        # PROJECTION HEAD
        if self.project_head.training:
            for layer in self.project_head:
                x=layer(x)
        return x

