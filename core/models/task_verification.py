import torch
import torch.nn as nn

class TaskVerificationTransformer(nn.Module):
    def __init__(self, input_dim, num_heads=4, hidden_dim=256, num_layers=1):
        super().__init__()
        
        # We use a TransformerEncoderLayer instead of a DecoderLayer.
        # This allows the model to leverage bidirectional self-attention, 
        # looking at the entire recipe sequence (past and future steps) at once.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # A simple linear head to project the pooled sequence into a single binary logit
        self.classifier = nn.Linear(input_dim, 1)
        
    def forward(self, x, mask=None):
        # x shape: (Batch_Size, Max_Sequence_Length, Input_Dimension)
        # mask shape: (Batch_Size, Max_Sequence_Length) - True for padded positions
        out = self.transformer(x, src_key_padding_mask=mask)
        
        if mask is not None:
            # Invert the mask: 1.0 for valid tokens, 0.0 for padding tokens
            active_tokens = (~mask).unsqueeze(-1).float()
            
            # Zero-out the padded outputs and sum along the sequence dimension
            sum_out = (out * active_tokens).sum(dim=1)
            
            # Count the actual number of valid tokens per recipe
            count = active_tokens.sum(dim=1).clamp(min=1e-9)
            pooled_out = sum_out / count
        else:
            pooled_out = out.mean(dim=1)
            
        # Output shape: (Batch_Size)
        logits = self.classifier(pooled_out)
        return logits.squeeze(-1)