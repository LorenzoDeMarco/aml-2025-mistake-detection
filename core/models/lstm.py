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
        
        # Bi-LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if self.num_layers > 1 else 0
        )
        

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        # x shape: [Batch, Time, Features]
        x = torch.nan_to_num(x, nan=0.0)
        
        # lstm_out: [Batch, Time, Hidden*2]
        _, (h_n, _) = self.lstm(x)
        
        h_fwd = h_n[-2] 
        h_bwd = h_n[-1]
   
        combined_state = torch.cat([h_fwd, h_bwd], dim=-1)
        
        if combined_state.dim() == 1:
            combined_state = combined_state.unsqueeze(0)
            
        return self.classifier(combined_state)