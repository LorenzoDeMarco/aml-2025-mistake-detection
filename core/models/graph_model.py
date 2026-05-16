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
        seq_len = x.size(0)
        return x + self.pe[:seq_len, :]

class GraphClassifier(nn.Module):
    def __init__(self, visual_dim=768, text_dim=256, hidden_dim=256, num_layers=2, dropout_prob=0.4):
        super().__init__()
        
        self.matcher = StepMatchingModule(visual_dim, text_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(d_model=text_dim)
        
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
            visual_feats: (Num_Visual, 768)
            text_feats:   (Num_Nodes + 1, 256) # Includes Virtual Node
            adj_matrix:   (Num_Nodes + 1, Num_Nodes + 1)
        """
        # Exclude the virtual node from the visual matching process
        real_text_feats = text_feats[:-1, :]
        virtual_feat = text_feats[-1:, :]
        
        # Apply Positional Encoding ONLY to the real sequential steps
        real_text_feats = self.pos_encoder(real_text_feats)
        
        update_nodes, _, _ = self.matcher(visual_feats, real_text_feats)
        
        # Re-attach the virtual node to the updated graph
        x = torch.cat([update_nodes, virtual_feat], dim=0)
        
        num_total_nodes = x.size(0)
        
        # Message Passing Loop
        for i, layer in enumerate(self.gnn_layers):
            support = layer(x)
            
            # Create feature pairs for attention (src || dst)
            # Shape (N,N,2*F)
            x_i = support.unsqueeze(1).expand(-1, num_total_nodes, -1)
            x_j = support.unsqueeze(0).expand(num_total_nodes, -1, -1)
            attention_input = torch.cat([x_i, x_j], dim=-1)
            
            # Calculate attention scores 
            attention_scores = self.attention_layers[i](attention_input).squeeze()
            attention_scores = F.leaky_relu(attention_scores, negative_slope=0.2)
            
            # Mask out non-existent edges using the adjacency matrix (-inf for exp)
            zero_vec = -9e15 * torch.ones_like(attention_scores)
            attention_scores = torch.where(adj_matrix > 0, attention_scores, zero_vec)
            
            # Normalize attention scores with softmax
            attention_weights = F.softmax(attention_scores, dim=-1)
            
            # Aggregate neighbors using attention weights instead of uniform mean 
            x_aggregated = torch.matmul(attention_weights, support)
            
            x = F.relu(x_aggregated)
            x = self.norm(x)
            x = self.dropout(x)
             
        # Global Graph Readout via Virtual Node
        # Instead of Max/Mean pooling, extract the feature vector of the Virtual Node
        # which has automatically collected global context from the entire recipe.
        graph_rep = x[-1] 
        
        return self.classifier(graph_rep)