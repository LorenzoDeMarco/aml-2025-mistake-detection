import torch
import torch.nn as nn 
import torch.nn.functional as F 
from core.models.step_matching import StepMatchingModule
import math

class DAGNNLayer(nn.Module):
    """
    Custom Implementation of a Directed Acyclic Graph Neural Network Layer.
    Uses a GRU-like gating mechanism to update node states based on predecessors.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        
        # Projection for incoming messages 
        self.message_weight = nn.Linear(hidden_dim, hidden_dim)
        
        # GRU Gates (Input size: hidden_dim message + hidden_dim state = hidden_dim * 2)
        self.update_gate = nn.Linear(hidden_dim*2, hidden_dim)
        self.reset_gate = nn.Linear(hidden_dim*2, hidden_dim)
        self.cell_gate = nn.Linear(hidden_dim*2, hidden_dim)
        
    def forward(self, x, adj_matrix):
        # x shape: (Batch_Size, Num_Nodes, Feature_dim)
        # adj_matrix shape: (Batch_Size, Num_Nodes, Num_Nodes)
        
        # Generate messages for all nodes
        messages = self.message_weight(x)
        
        # Aggregate messagges strictly from predecessors
        # (adj_metrix MUST be lower-triangular / directional)
        agg_messages = torch.bmm(adj_matrix, messages)
        
        # Gated update (GRU mecchanism)
        cat_state = torch.cat([x, agg_messages], dim=-1)
        
        # Compute graph
        z = torch.sigmoid(self.update_gate(cat_state)) # what to keep
        r = torch.sigmoid(self.reset_gate(cat_state)) # what to forget
        
        # Compute candidate hidden state
        cat_reset_state = torch.cat([x*r, agg_messages], dim=-1)
        h_tilde = torch.tanh(self.cell_gate(cat_reset_state))
        
        # Final node update 
        return (1-z)*x + z*h_tilde

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
        
        # DAGNN
        self.dagnn_layers = nn.ModuleList([
            DAGNNLayer(hidden_dim=text_dim) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(text_dim)
        self.dropout = nn.Dropout(p=dropout_prob)
        
        self.attention = nn.Sequential(
            nn.Linear(text_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
         
        self.classifier = nn.Sequential(
            nn.Linear(text_dim * 2, hidden_dim),
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
        
        # Compute valid lengths for the entire batch simultaneously via boolean masking
        v_lens = (visual_feats != 0).any(dim=-1).sum(dim=-1).tolist()
        t_lens = (real_text_feats != 0).any(dim=-1).sum(dim=-1).tolist()
        
        # Perform fully batched Hungarian Matching
        updated_real_text = self.matcher(visual_feats, real_text_feats, v_lens, t_lens)
        
        # Re-assemble graph sequence
        x = torch.cat([updated_real_text, virtual_feats], dim=1)
        
        # DAGNN message passing loop
        for layer in self.dagnn_layers:
            x_residual = x
            x_new = layer(x, adj_matrix)
            x_new = self.norm(x_new)
            x_new = self.dropout(x_new)
            x = x_residual + x_new
            
        # Rich readout 
        virtual_rep = x[:, -1, :]
        real_nodes = x[:, :-1, :]
        
        # Compute raw attention scores
        attn_scores = self.attention(real_nodes).squeeze(-1)  # (Batch_Size, Max_Nodes)
        
        # Dynamic padding masking to prevent gradient updates on dummy/padded nodes
        # Identify real nodes that consist entirely of zero padding
        padding_mask = (real_nodes == 0).all(dim=-1)  # (Batch_Size, Max_Nodes)
        
        # Assign a strongly negative value (-10000.0 is safe for FP16/AMP numeric limits) to padded positions
        attn_scores = attn_scores.masked_fill(padding_mask, -10000.0)
        
        # Softmax activation to derive the probability distribution exclusively over valid sequence nodes
        attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)  # (Batch_Size, Max_Nodes, 1)
        
        # Weighted summation (Content-dependent Attention Pooling instead of arithmetic mean)
        attended_pool_rep = torch.sum(real_nodes * attn_weights, dim=1)  # (Batch_Size, text_dim)
        
        # Final feature concatenation to preserve the downstream classifier input dimension (text_dim * 2)
        graph_rep = torch.cat([virtual_rep, attended_pool_rep], dim=-1)
        logits = self.classifier(graph_rep)
        
        return logits.squeeze(-1)