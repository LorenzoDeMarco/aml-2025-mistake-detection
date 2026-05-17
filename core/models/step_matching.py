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
        
    def forward(self, batched_visual, batched_text, v_lens, t_lens):
        """
        Args:
            visual_feats: Tensor of shape (num_visual_steps, 768)
            text_feats: Tensor of shape (num_graph_nodes, 256)
        Returns:
            updated_nodes: Tensor of shape (num_graph nodes, 256)
        """
        batch_size = batched_visual.size(0)
        
        # Project all visual features in parallel on GPU
        proj_visual = self.visual_proj(batched_visual)
        
        # Normalize features to compute Cosine Similarity efficiently via dot product
        norm_visual = F.normalize(proj_visual, p=2, dim=1)
        norm_text = F.normalize(batched_text, p=2, dim=1)
        
        # Batched Matrix Multiplication to compute all similarities at once (GPU)
        # Result shape: (Batch, Max_Visual_Steps, Max_Nodes)
        sim_matrix = torch.bmm(norm_visual, norm_text.transpose(1, 2))
        
        # Move the ENTIRE batch of cost matrices to CPU exactly ONCE
        cost_matrices = (1.0 - sim_matrix).detach().cpu().numpy()
        
        updated_nodes_list = []
        for b in range(batch_size):
            v_len = v_lens[b]
            t_len = t_lens[b]
            
            # Extract only the valid, unpadded portion of the cost matrix
            valid_cost = cost_matrices[b,  :v_len, :t_len]
            
            # Hungarian matching (CPU)
            row_idx, col_idx = linear_sum_assignment(valid_cost)
            
            # Feature update (GPU)
            updated_t = batched_text[b, :t_len, :].clone()
            
            # Grather matching features using indexing to preserve backpropagation gradients
            matched_v = proj_visual[b, row_idx, :]
            matched_t = proj_visual[b, col_idx, :]
            
            combined_feat = torch.cat([matched_v, matched_t], dim=1)
            fused_feat = self.fusion_proj(combined_feat)
            
            updated_t[col_idx] = fused_feat.to(updated_t.dtype)
            
            # Pad back to max dimensions to mantain 3D batch shape
            padded_updated = torch.zeros_like(batched_text[b])
            padded_updated[:t_len, :] = updated_t
            updated_nodes_list.append(padded_updated)
            
        return torch.stack(updated_nodes_list, dim=0)