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
            text_feats: (Num_Nodes, 256)
            adj_matrix: (Num_Nodes, Num_Nodes) - 1 where edge exists
        """
        
        # Cross-modal features update
        update_nodes, _, _ = self.matcher(visual_feats, text_feats)
        
        x = update_nodes
        for layer in self.gnn_layers:
            support = layer(x)
            
            x_aggregated = torch.matmul(adj_matrix, support)
            
            # Apply activation, normalization, and dropout
            x = F.relu(x_aggregated)
            x = self.norm(x)
            x = self.dropout(x)
            
        # Global Graph Readout (Max Pooling to capture localized anomalies)
        graph_rep, _ = torch.max(x, dim=0)
        
        # Final Classification
        logits = self.classifier(graph_rep)
        
        return logits