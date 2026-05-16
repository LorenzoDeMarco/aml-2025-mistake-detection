import torch
import torch.nn as nn
import math

class TaskVerificationTransformer(nn.Module):
    def __init__(self, input_dim=768, embed_dim=256, num_heads=8, num_layers=4, dropout=0.3, max_seq_len=1050):
        super(TaskVerificationTransformer, self).__init__()
        
        # 1.projection layer to reduce dimensionality from 768 to embed_dim
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
        # 2.positional encoding to inject sequence order information (important for task verification)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout, max_len=max_seq_len)
        
        # 3. transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4, 
            dropout=dropout,
            batch_first=True # [B, N, D] batch size, sequence leght (fixed at max_seq_len), embedding dimension 356 post linear projection
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. classification head (MLP)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1) #binary classification so single logit output
        )
        
    def forward(self, x, mask=None):
        """
        x: [Batch_Size, Max_Seq_Len, 768]
        mask: [Batch_Size, Max_Seq_Len] (1 per dati reali, 0 per padding)
        """

        x = self.input_proj(x) # [B, N, embed_dim]
        x = self.pos_encoder(x)
        
        #padding mask for transformer (True for positions that are padded and should be ignored)
        src_key_padding_mask = (mask == 0) if mask is not None else None
        
        output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # Global Average Pooling include only valid positions (mask == 1)
        if mask is not None:
            # Masked Mean Pooling
            mask_expanded = mask.unsqueeze(-1).expand_as(output)
            sum_embeddings = torch.sum(output * mask_expanded, dim=1)
            mean_pooled = sum_embeddings / torch.clamp(torch.sum(mask, dim=1, keepdim=True), min=1e-9)
        else:
            mean_pooled = torch.mean(output, dim=1)
            
        # final classification 
        logits = self.classifier(mean_pooled)
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