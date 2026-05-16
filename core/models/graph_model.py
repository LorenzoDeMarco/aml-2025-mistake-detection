import torch
import torch.nn as nn 
import torch.nn.functional as F 
from core.models.step_matching import StepMatchingModule
import math

class PositionalEncoding(nn.Module):
    """
    Injects information about the relative or absolute position of the recipe steps into the node features.
    """
    def __init__(self, d_model, max_len=100):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: (Batch_Size, Num_Real_Nodes, Feature_Dim)
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)

class GraphClassifier(nn.Module):
    def __init__(self, visual_dim=768, text_dim=256, hidden_dim=256, num_layers=2, dropout_prob=0.4):
        super().__init__()
        
        self.matcher = StepMatchingModule(visual_dim, text_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(d_model=text_dim)
        
        # Graph Attention Layers (GAT-like for 3D batched tensors)
        self.attention_layers = nn.ModuleList([
            nn.Linear(text_dim * 2, 1) for _ in range(num_layers)
        ])
        
        self.gnn_layers = nn.ModuleList([
            nn.Linear(text_dim , text_dim) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(text_dim)
        self.dropout = nn.Dropout(p=dropout_prob)
         
        self.classifier = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, visual_feats, text_feats, adj_matrix):
        """
        Args:
            visual_feats: (Batch_Size, Max_Visual_Steps, 768)
            text_feats:   (Batch_Size, Max_Nodes + 1, 256) # Includes Virtual Node
            adj_matrix:   (Batch_Size, Max_Nodes + 1, Max_Nodes + 1)
        """
        batch_size = text_feats.size(0)
        num_total_nodes = text_feats.size(1)
        
        # Slicing out real features and the virtual node tracking the absolute index (-1)
        real_text_feats = text_feats[:, :-1, :]
        virtual_feats = text_feats[:, -1:, :]
        
        # Positional Encoding on real sequence items
        real_text_feats = self.pos_encoder(real_text_feats)
        
        # Contextual alignment loop to preserve compatibility with 2D matching modules
        updated_real_list = []
        for b in range(batch_size):
            # Dynamic unpadding: identify the valid length of sequences for this specific graph
            # We count rows that contain at least one non-zero element
            v_len = (visual_feats[b] != 0).any(dim=-1).sum()
            t_len = (real_text_feats[b] != 0).any(dim=-1).sum()
            
            # Extract exclusively the real nodes to prevent Hungarian Matching from aligning with padding zeros
            real_v = visual_feats[b, :v_len, :]
            real_t = real_text_feats[b, :t_len, :]
            
            # Perform matching safely
            updated_real_t, _, _ = self.matcher(real_v, real_t)
            
            # Re-assemble the tensor with padding to maintain 3D batch structure
            padded_updated = torch.zeros_like(real_text_feats[b])
            padded_updated[:t_len, :] = updated_real_t
            updated_real_list.append(padded_updated)
            
        update_nodes = torch.stack(updated_real_list, dim=0)
        
        # Re-assemble graph sequence
        x = torch.cat([update_nodes, virtual_feats], dim=1)
        
        # Message Passing Loop
        for i, layer in enumerate(self.gnn_layers):
            support = layer(x) # Shape: (Batch_Size, N, F)
            
            # Broadcast token pairs to evaluate adjacency weights: (Batch_Size, N, N, 2*F)
            x_i = support.unsqueeze(2).expand(-1, -1, num_total_nodes, -1)
            x_j = support.unsqueeze(1).expand(-1, num_total_nodes, -1, -1)
            attention_input = torch.cat([x_i, x_j], dim=-1)
            
            attention_scores = self.attention_layers[i](attention_input).squeeze(-1)
            attention_scores = F.leaky_relu(attention_scores, negative_slope=0.2)
            
            # CRITICAL MASKING step: Set dummy positions and non-edges to -inf
            # Padded entries have an adjacency coordinate of 0.0, filtering them out instantly
            zero_mask = -9e15 * torch.ones_like(attention_scores)
            attention_scores = torch.where(adj_matrix > 0, attention_scores, zero_mask)
            
            # Softmax assigns exactly 0.0 weight to the masked fictitious nodes
            attention_weights = F.softmax(attention_scores, dim=-1)
            
            # Batched Matrix Multiplication to route the representations
            x_aggregated = torch.bmm(attention_weights, support)
            
            x = F.relu(x_aggregated)
            x = self.norm(x)
            x = self.dropout(x)
            
        # Global Graph Readout via Virtual Node (always safe at index -1)
        graph_rep = x[:, -1, :] 
        
        logits = self.classifier(graph_rep)
        return logits.squeeze(-1)