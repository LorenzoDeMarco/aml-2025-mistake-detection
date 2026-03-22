import math
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
from constants import Constants as const
# define the transformer backbone here
EncoderLayer = nn.TransformerEncoderLayer
Encoder = nn.TransformerEncoder


def fetch_input_dim(config, decoder=False):
    if config.backbone == const.OMNIVORE:
        return 1024
    elif config.backbone == const.SLOWFAST:
        return 400
    elif config.backbone == const.X3D:
        return 400
    elif config.backbone == const.RESNET3D:
        return 400
    elif config.backbone == const.IMAGEBIND:
        if decoder is True:
            return 1024
        k = len(config.modality)
        return 1024 * k
    elif config.backbone == const.PERCEPTION_ENCODER:
        # PE-Core-B16：CLIP Dim = 1024
        # https://huggingface.co/facebook/PE-Core-L14-336
        return 1024

    # ❗ Important: never silently return None
    raise ValueError(f"Unknown backbone for fetch_input_dim: {config.backbone}")


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x


class MLP1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP1, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size * 8)
        self.layer2 = nn.Linear(hidden_size * 8, hidden_size * 2)
        self.layer3 = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class CNN(nn.Module):
    def __init__(self, in_channels, final_width, final_height, num_classes):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * final_width * final_height, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor, indices=None) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        if indices is None:
            x = x + self.pe[:, :x.size(1)]
        else:
            pos = torch.cat([self.pe[:, index] for index in indices])
            x = x + pos
        return self.dropout(x)


class PositionalEncodingLearn(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.embed = nn.Embedding(max_len, d_model)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.embed.weight)

    def forward(self, x, indices=None):
        # x: b, l, d
        r = torch.arange(x.shape[1], device=x.device)
        embed = self.embed(r)  # seq_len, embedding_dim
        return x + embed.repeat(x.shape[0], 1, 1)

#setup2: add LSTM model Baseline
class LSTM(nn.Module):
    """
    LSTM baseline for step-level classification on pre-extracted features
    resource:https://github.com/ritchieng/deep-learning-wizard/blob/master/docs/deep_learning/practical_pytorch/pytorch_lstm_neuralnetwork.md
    https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.rnn.LSTM.html
    
    Parameters
    ----------
    input_dim : int
        Feature dimension D of each 1s sub-segment vector.
        MUST match  .npz feature dimension ( 1024 ).
    hidden_dim : int, default=256
        LSTM hidden state size.
    layer_dim : int, default=1
        Number of stacked LSTM layers. 
        Note: nn.LSTM's internal dropout is only applied when layer_dim > 1.
    bidirectional : bool, default=False
        If True, uses BiLSTM (often slightly stronger but a bit heavier).
    dropout : float, default=0.0
        Dropout probability between LSTM layers (only effective when layer_dim > 1).
    batch_first: If True, then the input and output tensors are provided as (batch, seq, feature):Default: False
    
    Forward Inputs
    --------------
    x : torch.Tensor
        Shape (B, T, D) if batch_first=True.
        B=batch size, T=number of sub-segments in a step, 1s or 10s,depend on the feature , D=input_dim.

    Returns
    -------
    logits: (B,T,1).
    """
    def __init__(self,input_size: int,  hidden_size: int = 256, num_layers: int = 2,dropout: float = 0.2):
        super().__init__()
        #Build a bidirectional LSTM encoder (expects inputs as (B, T, D)).
        self.lstm = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        # due to bidirectional=True
        # Compute the per-timestep output dimension (2H for BiLSTM).
        out_dim = hidden_size * 2 
        # Define a simple prediction head: normalize features, then project to 1 logit per timestep.
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
            nn.Linear(out_dim, 1),
        )

    def forward(self, x):
        # x: [T, D] or [B, T, D] 
        # Accept either a single sequence (T, D) or a batch (B, T, D).
        
        #Create a flag to remember whether we added a batch dimension.
        squeeze = False
        
        #If a single sequence is provided, add a batch dimension to make it (1, T, D).
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1, T, D]
            squeeze = True

        # Run the BiLSTM to produce contextual features y for each timestep (B, T, 2H). 
        # Ignore (h_n, c_n) (the hidden state and memory unit), as it is not needed here.
        y, _ = self.lstm(x)        
        #Apply the head to get 1 raw logit per timestep (B, T, 1).
        logits = self.head(y)       
        #Remove the temporary batch dimension for single-sequence input; otherwise return batched logits as-is.
        if squeeze:
            return logits.squeeze(0)
        else:
            return logits