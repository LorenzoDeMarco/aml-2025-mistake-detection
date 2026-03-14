import torch
from torch import nn
from core.models.blocks import fetch_input_dim


class LSTMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        input_dim = fetch_input_dim(config)
        self.hidden_dim = 256
        self.num_layers = 2
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x shape: [Batch, Time, Feat] 
        x = torch.nan_to_num(x, nan=0.0)
        
        # lstm_out shape: [Batch, Time, Hidden*2]
        lstm_out, _ = self.lstm(x)
        
        # Output shape: [1, 36, 1]
        logits = self.classifier(lstm_out)
 
        return logits.view(-1, 1) 