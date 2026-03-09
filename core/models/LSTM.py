import torch
from torch import nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        
        # Number of hidden layers
        self.layer_dim = layer_dim

        self.output_dim = output_dim
        
        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Dropout layer
        self.dropout = nn.Dropout(0.2)
        
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # Handle 2D input (seq, features) by adding batch dimension
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension: (1, seq, features)
        
        batch_size = x.size(0)
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        
        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        
        # One time step
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        
        # Index hidden state of last time step
        #out.size() --> 512, 1024, 512
        # out[:, -1, :] --> 512, 1 --> just want last time step hidden states! 
        #out = self.fc(out[:, -1, :]) 
        # out.size() --> 512, 1
        #return out
        # Apply fc to all time steps (not just the last)
        out = self.fc(out)  # (batch_size, seq_len, output_dim)
        
        # Flatten to (batch_size * seq_len, output_dim) for compatibility
        return out.view(-1, self.output_dim)
        