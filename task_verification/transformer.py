import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TaskVerificationTransformer(nn.Module):
    def __init__(self, input_dim=768, embed_dim=256, num_heads=8, num_layers=4, dropout=0.3, max_seq_len=1050):
        super(TaskVerificationTransformer, self).__init__()
        
        # linear projection
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
        # stride-4 Conv1D to downsample time dimension by 4
        self.temporal_downsample = nn.Conv1d(
            in_channels=embed_dim, 
            out_channels=embed_dim, 
            kernel_size=4, 
            stride=4, 
            padding=0
        )
        
        # learnable [CLS] token parameter
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        
        # positional encoding
        downsampled_max_len = (max_seq_len // 4) + 50 ##stsride 4 reduces max length by quarter, add some buffer for safety
        self.pos_encoder = PositionalEncoding(embed_dim, dropout, max_len=downsampled_max_len)
        
        #transformer encoder layers with Pre-LN (norm_first=True) for deep stability
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4, 
            dropout=dropout,
            batch_first=True,
            norm_first=True
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
        batch_size = x.size(0)

        # project input features to embedding space
        x = self.input_proj(x) # [B, T, embed_dim]
        
        #apply Conv1D temporal downsampling
        x = x.permute(0, 2, 1) # [B, embed_dim, T]
        x = self.temporal_downsample(x) # [B, embed_dim, T_new]
        x = x.permute(0, 2, 1) # [B, T_new, embed_dim]
        
        #compress mask to match the downsampled features using MaxPool1D
        if mask is not None:
            mask = mask.unsqueeze(1) # [B, 1, T]
            mask = F.max_pool1d(mask, kernel_size=4, stride=4) # [B, 1, T_new]
            mask = mask.squeeze(1) # [B, T_new]
            
            #align padding mismatches due to integer division rounding
            if mask.size(1) > x.size(1):
                mask = mask[:, :x.size(1)]
            elif mask.size(1) < x.size(1):
                padding = torch.zeros((mask.size(0), x.size(1) - mask.size(1)), device=mask.device)
                mask = torch.cat([mask, padding], dim=1)
        
        #prrepend the learnable [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1) # [B, T_new + 1, embed_dim]
        
        x = self.pos_encoder(x)
        
        # [CLS] should not be masked
        if mask is not None:
            cls_mask = torch.ones((batch_size, 1), dtype=mask.dtype, device=mask.device)
            extended_mask = torch.cat((cls_mask, mask), dim=1) # [B, T_new + 1]
            src_key_padding_mask = (extended_mask == 0) # True means ignore padding position
        else:
            src_key_padding_mask = None
    
        output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask) # [B, T_new + 1, embed_dim]
        
        #extract the [CLS] token output for classification
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