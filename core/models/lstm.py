import torch
from torch import nn
from core.models.blocks import MLP, fetch_input_dim

class LSTMModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config= config
        input_dimension = fetch_input_dim(config)

        self.hidden_dim = getattr(config, 'lstm_hidden_dim', 256)
        self.num_layers = getattr(config, 'lstm_layers', 2)

        # Layer LSTM
        self.lstm= nn.LSTM(
            input_size=input_dimension,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if self.num_layers > 1 else 0.0
        )

        # Decoder MLP
        self.decoder = MLP(self.hidden_dim * 2, 512, 1)  # *2 for bidirectional

    def forward(self, x):
        # x shape: (batch, seq_len, feature_dim)
        x = torch.nan_to_num(x, nan=0.0)

        # lsto_out shape: (batch, seq_len, hidden_dim * 2)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Global Average Pooling over the sequence dimension
        encoded_output = torch.mean(lstm_out, dim=1)

        return self.decoder(encoded_output)