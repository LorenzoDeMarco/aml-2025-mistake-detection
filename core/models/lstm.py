import torch
from torch import nn
import numpy as np
from core.models.blocks import fetch_input_dim


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=1, dropout=0.3):
        super(LSTMModel, self).__init__()
        #bidirectional = True => double the outgoing hidden_dim
        self.lstm = nn.LSTM(input_size=input_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=num_layers, 
                            batch_first=True, 
                            bidirectional=True, 
                            dropout=dropout if num_layers > 1 else 0.0)
        #classifier takes 2 * hidden_dim due to bidirectionality
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1) ## Binary output for each frame
        )
    def forward(self, x):
        # x shape: (Batch=1, Seq_Len=N, Features=D)
        # since DataLoader currently returns (N, D), we add the batch dimension
        if x.dim() == 2:
            x = x.unsqueeze(0) 
            
        lstm_out, _ = self.lstm(x) # lstm_out shape: (Batch, Seq_Len, hidden_dim*2)
        logits = self.classifier(lstm_out) # logits shape: (Batch, Seq_Len, 1)
        # remove the batch dimension to match target shape (N, 1)
        return logits.squeeze(0)
