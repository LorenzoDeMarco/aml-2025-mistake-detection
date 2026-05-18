import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class GraphNodeRealizer(nn.Module):
    def __init__(self, visual_dim=768, text_dim=256, joint_dim=256, dropout=0.4):
        """
        Substep 3 Advanced Realizer: Features alignment with strict slicing,
        differentiable auxiliary alignment loss, and cosine semantic thresholding.
        """
        super(GraphNodeRealizer, self).__init__()
        
        #Conv1d layer for stride-4 temporal compression of visual features
        self.temporal_conv = nn.Conv1d(
            in_channels=visual_dim, out_channels=visual_dim,
            kernel_size=4, stride=4, padding=0
        )
        
        # Projections designed for similarity calculation
        self.sim_visual_proj = nn.Linear(visual_dim, joint_dim)
        self.sim_text_proj = nn.Linear(text_dim, joint_dim)
        
        #dual-pathway specialized fusion expert heads
        self.matched_projection = nn.Sequential(
            nn.Linear(visual_dim + text_dim, joint_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(joint_dim * 2, joint_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(joint_dim)
        )
        
        self.unmatched_projection = nn.Sequential(
            nn.Linear(text_dim, joint_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(joint_dim)
        )
        
        #learnable missing execution signifier
        self.missing_visual_embedding = nn.Parameter(torch.zeros(1, visual_dim))
        nn.init.normal_(self.missing_visual_embedding, std=0.02)

    def forward(self, visual_features, text_features, visual_mask, text_mask):
        batch_size, max_m, text_dim = text_features.shape
        device = visual_features.device
        
        # stride-4 temporal compression
        vis_transposed = visual_features.transpose(1, 2)
        compressed_vis = self.temporal_conv(vis_transposed).transpose(1, 2)
        compressed_mask = visual_mask[:, ::4][:, :compressed_vis.size(1)]
        comp_n = compressed_vis.size(1)
        
        proj_v = self.sim_visual_proj(compressed_vis) # [B, Comp_N, 256]
        proj_t = self.sim_text_proj(text_features)     # [B, Max_M, 256]
        
        realized_nodes = torch.zeros(batch_size, max_m, self.sim_text_proj.out_features, device=device)
        extended_visual = torch.cat([compressed_vis, self.missing_visual_embedding.expand(batch_size, 1, -1)], dim=1)
        
        aux_alignment_loss = torch.tensor(0.0, device=device)
        valid_batch_counts = 0
        
        for b in range(batch_size):
            num_vis = int(compressed_mask[b].sum().item())
            num_text = int(text_mask[b].sum().item())
            
            #default lookup arrays tracking mapping positions (Fallback to comp_n)
            node_mapping = torch.full((max_m,), fill_value=comp_n, dtype=torch.long, device=device)
            is_matched_node = torch.zeros(max_m, dtype=torch.bool, device=device)
            
            if num_vis > 0 and num_text > 0:
                #strict slicing: isolate valid elements to bypass padding items completely
                v_norm = F.normalize(proj_v[b, :num_vis], p=2, dim=-1)
                t_norm = F.normalize(proj_t[b, :num_text], p=2, dim=-1)
                
                #differentiable similarity computation sub-graph
                similarity_matrix = torch.mm(v_norm, t_norm.t()) # [num_vis, num_text]
                
                cost_matrix = 1.0 - similarity_matrix.detach().cpu().numpy()
                v_idx, t_idx = linear_sum_assignment(cost_matrix)
                
                if len(v_idx) > 0:
                    #filter forced poor assignments via Cosine Semantic Thresholding (0.20)
                    matched_sims = similarity_matrix[v_idx, t_idx]
                    
                    #accumulate differentiable similarity signal to update projection layers
                    aux_alignment_loss += -matched_sims.mean()
                    valid_batch_counts += 1
                    
                    for v_i, t_i in zip(v_idx, t_idx):
                        if similarity_matrix[v_i, t_i] >= 0.20: #semantic sanity threshold
                            node_mapping[t_i] = v_i
                            is_matched_node[t_i] = True
            
            gathered_vis = extended_visual[b, node_mapping]
            fused_raw_inputs = torch.cat([gathered_vis, text_features[b]], dim=-1)
            
            matched_out = self.matched_projection(fused_raw_inputs)
            unmatched_out = self.unmatched_projection(text_features[b])
            
            sample_realized = torch.where(is_matched_node.unsqueeze(-1), matched_out, unmatched_out)
            realized_nodes[b] = sample_realized * text_mask[b].unsqueeze(-1)
            
        if valid_batch_counts > 0:
            aux_alignment_loss = aux_alignment_loss / valid_batch_counts
            
        return realized_nodes, aux_alignment_loss