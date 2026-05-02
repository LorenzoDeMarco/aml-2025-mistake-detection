import torch
import torch.nn as nn

class TaskVerificationTransformer(nn.Module):
    def __init__(self, input_dim, num_heads=4, hidden_dim=256, num_layers=1):
        super().__init__()
        
        encoder_layer = nn.TransformerDecoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        out = self.transformer(x)
        
        pooled_out = out.mean(dim=1)
        
        logits = self.classifier(pooled_out)
        
        return logits.squeeze(-1)
    
