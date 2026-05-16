import torch
import torch.nn as nn 
import torch.nn.functional as F 
from scipy.optimize import linear_sum_assignment

class StepMatchingModule(nn.Module):
    def __init__(self, visual_dim=768, text_dim=256, hidden_dim=256):
        super().__init__()
        
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, text_dim)
        )
        
        self.fusion_proj = nn.Sequential(
            nn.Linear(text_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, text_dim)
        )
        
    @torch.compiler.disable
    def forward(self, visual_feats, text_feats):
        """
        Args:
            visual_feats: Tensor of shape (num_visual_steps, 768)
            text_feats: Tensor of shape (num_graph_nodes, 256)
        Returns:
            updated_nodes: Tensor of shape (num_graph nodes, 256)
        """
        
        # Project visual features to match the dimensionality of text features
        proj_visual = self.visual_proj(visual_feats)
        
        # Normalize features to compute Cosine Similarity efficiently via dot product
        norm_visual = F.normalize(proj_visual, p=2, dim=1)
        norm_text = F.normalize(text_feats, p=2, dim=1)
        
        # Compute similarity matrix. Shape: (num_visual_steps, num_graph_nodes)
        sim_matrix = torch.matmul(norm_visual, norm_text.t())
        
        # Convert similarity to cost (Hungarian algorithm minimizes cost)
        cost_matrix = (1.0 - sim_matrix).detach().cpu().numpy()
        
        # Hungarian matching 
        # linear_sum_assignment finds the optimal 1-to-1 bipartite matching
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        
        # Node feature update 
        # Clone the original text features to initialize the updated nodes
        updated_nodes = text_feats.clone()
        
        for v_idx, n_idx in zip(row_idx, col_idx):
            # Concatenate the projected visual feature with the corresponding text node
            combined_feat = torch.cat([proj_visual[v_idx], text_feats[n_idx]], dim=0)
            
            # Pass trough the learnable fusion projection to update the node
            updated_nodes[n_idx] = self.fusion_proj(combined_feat)
            
        return updated_nodes, row_idx, col_idx