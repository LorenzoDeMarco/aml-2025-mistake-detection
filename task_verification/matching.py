import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class GraphNodeRealizer(nn.Module):
    def __init__(self, visual_dim=768, text_dim=256, joint_dim=256, dropout=0.4):
        """
        Features compression, Hungarian matching, and node realization.
        Fuses 768 visual macro-tokens with 256 fine-grained mistake text dimensions.
        """
        super(GraphNodeRealizer, self).__init__()
        
        #temporal compressor: contracts time by 4x, maintains 768 structural dimensions
        self.temporal_conv = nn.Conv1d(
            in_channels=visual_dim,
            out_channels=visual_dim,
            kernel_size=4,
            stride=4,
            padding=0
        )
        
        # shared projections for cosine similarity matching
        self.sim_visual_proj = nn.Linear(visual_dim, joint_dim)
        self.sim_text_proj = nn.Linear(text_dim, joint_dim)
        
        #learnable fusion head for valid pairs: [Visual(768) || Text_Error(256)] -> [1024] -> [256]
        self.matched_projection = nn.Sequential(
            nn.Linear(visual_dim + text_dim, joint_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(joint_dim * 2, joint_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(joint_dim)
        )
        
        #learnable projection for unmatched nodes (Omissions path)
        self.unmatched_projection = nn.Sequential(
            nn.Linear(text_dim, joint_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(joint_dim)
        )
        
        #learnable missing Visual Token: learnable parameter representing missing execution nodes
        self.missing_visual_embedding = nn.Parameter(torch.zeros(1, visual_dim))
        nn.init.normal_(self.missing_visual_embedding, std=0.02)

    def forward(self, visual_features, text_features, visual_mask, text_mask):
        """
        Args:
            visual_features (Tensor): [B, Max_N_Steps, 768]
            text_features (Tensor): [B, Max_M_Nodes, 256]
            visual_mask (Tensor): [B, Max_N_Steps]
            text_mask (Tensor): [B, Max_M_Nodes]
        Returns:
            realized_nodes (Tensor): [B, Max_M_Nodes, 256] node states ready for GNN layers
        """
        batch_size, max_m, text_dim = text_features.shape
        device = visual_features.device
        
        #Conv1D, shape shuffle for Conv1d: [B, 768, Max_N_Steps]
        vis_transposed = visual_features.transpose(1, 2)
        compressed_vis = self.temporal_conv(vis_transposed)
        compressed_vis = compressed_vis.transpose(1, 2) # Back to [B, Compressed_N, 768]
        
        #downsample the visual padding mask to align with the Conv1d stride (4x compression)
        compressed_mask = visual_mask[:, ::4][:, :compressed_vis.size(1)]
        comp_n = compressed_vis.size(1)
        
        #metric projections for cosine similarity matching
        proj_v = self.sim_visual_proj(compressed_vis) # [B, Comp_N, 256]
        proj_t = self.sim_text_proj(text_features)     # [B, Max_M, 256]
        
        realized_nodes = torch.zeros(batch_size, max_m, joint_dim := self.sim_text_proj.out_features, device=device)
        
        #append the learnable missing visual token at the end of each sample vector matrix to resolve unmatched indices
        missing_tokens = self.missing_visual_embedding.expand(batch_size, 1, -1)
        extended_visual = torch.cat([compressed_vis, missing_tokens], dim=1) # [B, Comp_N + 1, 768]
        
        #hungarian matching loop
        for b in range(batch_size):
            v_norm = F.normalize(proj_v[b], p=2, dim=-1)
            t_norm = F.normalize(proj_t[b], p=2, dim=-1)
            
            #cross similarity score mapping matrix: [Comp_N, Max_M]
            similarity_matrix = torch.mm(v_norm, t_norm.t())
            
            #invalidate visual padding tokens using our compressed binary mask
            v_mask_b = compressed_mask[b].unsqueeze(1)
            similarity_matrix = similarity_matrix * v_mask_b + (1.0 - v_mask_b) * -1.0
            
            #invalidate textual padding nodes using the text mask
            t_mask_b = text_mask[b].unsqueeze(0)
            similarity_matrix = similarity_matrix * t_mask_b + (1.0 - t_mask_b) * -1.0
            
            #compute cost matrix and optimize sum assignments
            cost_matrix = 1.0 - similarity_matrix.detach().cpu().numpy()
            v_idx, t_idx = linear_sum_assignment(cost_matrix)
            
            #initialize map indices pointing to the missing token index (Comp_N) as default
            node_mapping = torch.full((max_m,), fill_value=comp_n, dtype=torch.long, device=device)
            if len(t_idx) > 0:
                node_mapping[t_idx] = torch.tensor(v_idx, dtype=torch.long, device=device)
                
            #node feature extraction
            #gather associated visual macro-tokens (or missing parameter if node_mapping == comp_n)
            gathered_vis = extended_visual[b, node_mapping] # [Max_M, 768]
            
            #fusion layer concatenation: [Max_M, 768 + 256] -> [Max_M, 1024]
            fused_raw_inputs = torch.cat([gathered_vis, text_features[b]], dim=-1)
            
            #separate active nodes from padding nodes using boolean masking
            is_matched_node = torch.zeros(max_m, dtype=torch.bool, device=device)
            if len(t_idx) > 0:
                is_matched_node[t_idx] = True
            is_matched_node = is_matched_node.unsqueeze(-1) # [Max_M, 1]
            
            #expert projection modules
            matched_out = self.matched_projection(fused_raw_inputs)
            unmatched_out = self.unmatched_projection(text_features[b])
            
            #apply valid text nodes tracking mask to clean out batch padding nodes
            sample_realized = torch.where(is_matched_node, matched_out, unmatched_out)
            valid_nodes_mask = text_mask[b].unsqueeze(-1)
            realized_nodes[b] = sample_realized * valid_nodes_mask
            
        return realized_nodes