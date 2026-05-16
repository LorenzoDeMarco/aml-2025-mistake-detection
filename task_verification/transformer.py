import torch
import torch.nn as nn
import math

class TaskVerificationTransformer(nn.Module):
    def __init__(self, input_dim=768, embed_dim=256, num_heads=8, num_layers=4, dropout=0.3, max_seq_len=1050):
        super(TaskVerificationTransformer, self).__init__()
        
        self.max_seq_len = max_seq_len
        
        # projection layer to reduce dimensionality from 768 to embed_dim
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
        # learnable [CLS] token parameter (like in Bert, initialized randomly, optimized during training)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        
        # positional encoding to inject sequence order information
        self.pos_encoder = PositionalEncoding(embed_dim, dropout, max_len=max_seq_len)
        
        # transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4, 
            dropout=dropout,
            batch_first=True,
            norm_first=True # pre-layer normalization for deeper training stability with long sequences
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # classification head (MLP)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1) 
        )
        
    def forward(self, x, mask=None):
        """
        x: [Batch_Size, Current_Batch_Max_Len, 768]
        mask: [Batch_Size, Current_Batch_Max_Len] (1.0 for real data, 0.0 for padding)
        """
        batch_size = x.size(0)

        x = self.input_proj(x) # [B, N, embed_dim]
        
        # prepend the [CLS] token to the sequence
        # expand CLS token to match the current batch size: [B, 1, embed_dim]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # new sequence length becomes N + 1
        x = torch.cat((cls_tokens, x), dim=1) # [B, N + 1, embed_dim]
        
        #applyy positional encoding (it dynamically handles N + 1 positions up to max_seq_len)
        x = self.pos_encoder(x)
        
        # adjust the attention mask to account for the prepended [CLS] token that should not be masked
        if mask is not None:
            cls_mask = torch.ones((batch_size, 1), dtype=mask.dtype, device=mask.device)
            extended_mask = torch.cat((cls_mask, mask), dim=1) # [B, N + 1]
            src_key_padding_mask = (extended_mask == 0)
        else:
            src_key_padding_mask = None
        
        output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask) # [B, N + 1, embed_dim]
        
        #output from the [CLS] token 
        cls_output = output[:, 0, :] # [B, embed_dim]
            
        logits = self.classifier(cls_output)
        return logits.squeeze(-1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)