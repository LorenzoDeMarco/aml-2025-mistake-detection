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
        self.num_layers = getattr(config, 'num_layers', 2)
        
        # Linear layer to project input to hidden dimension for residual connection
        self.residual_proj = nn.Linear(input_dim, self.hidden_dim * 2)
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if self.num_layers > 1 else 0
        )
        
        self.layer_norm = nn.LayerNorm(self.hidden_dim * 2)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1)
        )
        
        #bias initialization to handle class imbalance
        prior = 0.3
        bias_value = -np.log((1 - prior) / prior)
        nn.init.constant_(self.classifier[-1].bias, bias_value)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        x = torch.nan_to_num(x, nan=0.0)
        
        #residual connection
        res = self.residual_proj(x)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # residual sum plus layer norm
        out = self.layer_norm(lstm_out + res)
        
        # Classification Many-to-Many
        logits = self.classifier(out)
        
        return logits.squeeze(0)