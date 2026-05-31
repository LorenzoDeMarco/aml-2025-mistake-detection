import torch
from torch import nn
import numpy as np
from core.models.blocks import fetch_input_dim


class LSTMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        input_dim = fetch_input_dim(config)

        self.hidden_dim = getattr(config, 'hidden_dim', 256)
        self.num_layers = getattr(config, 'num_layers', 1)

        #bidirectional = True => double the outgoing hidden_dim
        self.lstm = nn.LSTM(input_size=input_dim, 
                            hidden_size=self.hidden_dim, 
                            num_layers=self.num_layers, 
                            batch_first=True, 
                            bidirectional=True, 
                            dropout=0.3 if self.num_layers > 1 else 0)
        
        #classifier takes 2 * hidden_dim due to bidirectionality
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.lstm_dropout),
            nn.Linear(self.hidden_dim, 1) ## Binary output for each frame
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
