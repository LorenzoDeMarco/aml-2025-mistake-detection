import torch
import torch.nn as nn 
import torch.nn.functional as F 
from core.models.step_matching import StepMatchingModule

class GraphClassifier(nn.Module):
    def __init__(self, visual_dim=768, text_dim=256, hidden_dim=256, num_layers=2, dropout_prob=0.4):
        super().__init__()
        
        self.matcher = StepMatchingModule(visual_dim, text_dim, hidden_dim)
        
        self.gnn_layers = nn.ModuleList([
            nn.Linear(text_dim, text_dim) for _ in range(num_layers)
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
        
        update_nodes, _, _ = self.matcher(visual_feats, real_text_feats)
        
        # Re-attach the virtual node to the updated graph
        x = torch.cat([update_nodes, virtual_feat], dim=0)
        
        # Prevents feature explosion for nodes with many connections (like the Virtual Node)
        degree = adj_matrix.sum(dim=-1, keepdim=True)
        adj_normalized = adj_matrix / degree
        
        # Message Passing Loop
        for layer in self.gnn_layers:
            support = layer(x)
            
            # Use the normalized adjacency matrix for routing
            x_aggregated = torch.matmul(adj_normalized, support)
            
            x = F.relu(x_aggregated)
            x = self.norm(x)
            x = self.dropout(x)
            
        # Global Graph Readout via Virtual Node
        # Instead of Max/Mean pooling, extract the feature vector of the Virtual Node
        # which has automatically collected global context from the entire recipe.
        graph_rep = x[-1] 
        
        logits = self.classifier(graph_rep)
        return logits