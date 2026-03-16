
import torch
from torch import nn
import torch.nn.functional as F
from core.models.blocks import fetch_input_dim


class LSTMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        input_dim = fetch_input_dim(config)
        
        # config.py
        self.hidden_dim = getattr(config, 'hidden_dim', 256)
        self.num_layers = getattr(config, 'num_layers', 2)
        
        # Bi-LSTM: output_dim = hidden_dim * 2
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if self.num_layers > 1 else 0
        )
        
        self.layer_norm = nn.LayerNorm(self.hidden_dim * 2)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        
        # Mean pooling over the sequence dimension
        pooled_out = torch.mean(lstm_out, dim=1) 
        
        return self.classifier(pooled_out) 
