import torch
import torch.nn as nn 
import torch.nn.functional as F 
from core.models.step_matching import StepMatchingModule

class GraphClassifier(nn.Module):
    def __init__(self, visual_dim=768, text_dim=256, hidden_dim=256, num_layers=2):
        super().__init__()
        
        self.matcher = StepMatchingModule(visual_dim, text_dim, hidden_dim)
        
        self.gnn_layers = nn.ModuleList([
            nn.Linear(text_dim, text_dim) for _ in range(num_layers)
        ])
        
        self.classifier = nn.Sequential([
            nn.Linear(text_dim, text_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ])
        
    def forward(self, visual_feats, text_feats, adj_matrix):
        """
        Args:
            visual_feats: (Num_Visual, 768)
            text_feats: (Num_Nodes, 256)
            adj_matrix: (Num_Nodes, Num_Nodes) - 1 where edge exists
        """
        update_nodes, _, _ = self.matcher(visual_feats, text_feats)
        
        x = update_nodes
        for layer in self.gnn_layers:
            support = layer(x)
            x = F.relu(torch.matmul(adj_matrix, support) + x)
            
        graph_rep = x.mean(dim=0)
        
        logits = self.classifier(graph_rep)
        return logits.squeeze(-1)