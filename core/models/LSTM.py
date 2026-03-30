import torch
from torch import nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        dropout=0.5 
        
        # LSTM Bidirezionale
        # Nota: num_layers=layer_dim, bidirectional=True
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            layer_dim, 
            bidirectional=True, # usiamo una LSTM bidirezionale per migliorare le prestazioni
            batch_first=True,
            dropout=dropout
        )

        # Il livello Fully Connected riceve hidden_dim (forward only) o hidden_dim*2  (bidirezionale)
        self.fc = nn.Sequential( # Funnel Architecture
            nn.LayerNorm(hidden_dim * 2), # Normalizzazione per stabilizzare l'addestramento
            nn.Linear(hidden_dim *2, hidden_dim ), # Moltiplichiamo per 2 perché è bidirezionale
            nn.GELU(), # usiamo la GELU più performante della classica RELU 
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
            
        )
    
    def forward(self, x):
        # 1. Gestione input 2D (seq, features) -> (1, seq, features)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        #print(f"Input iniziale: {x.shape}")
        batch_size = x.size(0)
        device = x.device

        # 2. CORREZIONE: In una biLSTM, il numero di layer per h0/c0 deve essere:
        # layer_dim * 2 (uno per la direzione forward, uno per backward)
        num_directions = 2 if self.lstm.bidirectional else 1
        #h0 = torch.zeros(self.layer_dim * num_directions, batch_size, self.hidden_dim).to(device)
        #c0 = torch.zeros(self.layer_dim * num_directions, batch_size, self.hidden_dim).to(device)
        
        # 3. Passaggio nella LSTM
        # rimosso .detach() a meno che tu non faccia BPTT troncato manualmente
        out, _ = self.lstm(x)
        
        # out shape: (batch_size, seq_len, hidden_dim * 2)
        #print(f"Output LSTM shape: {out.shape}")
        
        # 4. Applicazione del readout layer a ogni istante temporale
        out = self.fc(out)  # shape: (batch_size, seq_len, output_dim)
        
        # 5. Ritorno in formato flat (batch, seq_len) se richiesto dalla loss
        #print(f"Output shape dopo FC: {out.shape}")
        #res= out.view(-1, self.output_dim)
       
        #print(f"Output shape after FC: {out.shape}, Reshape to: {res.shape}")
        #print(f"Output sample: {res[0]}")  # Stampa un campione per debug
        out = out.squeeze(0) # rimuove l'ultima dimensione se output_dim=1
        #print(f"Output finale dopo squeeze: {out.shape}")
        return  out
        