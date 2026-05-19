import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphNodeRealizer(nn.Module):
    def __init__(self, visual_dim=768, text_dim=256, joint_dim=256, dropout=0.4):
        """
        Video context token alignment with Hungarian Graph matching.
        """
        super(GraphNodeRealizer, self).__init__()
        
        # Conv1d (Stride 4 with padding)
        self.temporal_conv = nn.Conv1d(
            in_channels=visual_dim,
            out_channels=visual_dim,
            kernel_size=4,
            stride=4,
            padding=1
        )
        
        #metric space mapping projections
        self.sim_visual_proj = nn.Linear(visual_dim, joint_dim)
        self.sim_text_proj = nn.Linear(text_dim, joint_dim)
        
        #dual-pathway fusion experts heads
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
        
        #learnable missing visual token
        self.missing_visual_embedding = nn.Parameter(torch.zeros(1, visual_dim))
        nn.init.normal_(self.missing_visual_embedding, std=0.02)

    def forward(self, visual_features, text_features, visual_mask, text_mask, precomputed_matches):
        batch_size, max_m, text_dim = text_features.shape
        device = visual_features.device
        
        # dynamic temporal pooling step
        vis_transposed = visual_features.transpose(1, 2)
        compressed_vis = self.temporal_conv(vis_transposed).transpose(1, 2)
        comp_n = compressed_vis.size(1)
        
        # project modalities into common space
        proj_v = self.sim_visual_proj(compressed_vis)
        proj_t = self.sim_text_proj(text_features)
        
        realized_nodes = torch.zeros(batch_size, max_m, self.sim_text_proj.out_features, device=device)
        extended_visual = torch.cat([compressed_vis, self.missing_visual_embedding.expand(batch_size, 1, -1)], dim=1)
        
        aux_alignment_loss = torch.tensor(0.0, device=device)
        valid_batch_counts = 0
        
        for b in range(batch_size):

            match_indices = precomputed_matches[b]
            v_idx = match_indices[0]
            t_idx = match_indices[1]
            
            node_mapping = torch.full((max_m,), fill_value=comp_n, dtype=torch.long, device=device)
            is_matched_node = torch.zeros(max_m, dtype=torch.bool, device=device)
            
            if len(v_idx) > 0:
                #differentiable calculation of the cosine matrix on GPU
                v_norm = F.normalize(proj_v[b, v_idx], p=2, dim=-1)
                t_norm = F.normalize(proj_t[b, t_idx], p=2, dim=-1)
                matched_sims = (v_norm * t_norm).sum(dim=-1)
                
                aux_alignment_loss += -matched_sims.mean()
                valid_batch_counts += 1
                
                #dynamic threshold mask operations
                valid_threshold_mask = matched_sims >= 0.20
                valid_t = t_idx[valid_threshold_mask]
                valid_v = v_idx[valid_threshold_mask]
                
                if valid_t.numel() > 0:
                    node_mapping[valid_t] = valid_v
                    is_matched_node[valid_t] = True
                    
            gathered_vis = extended_visual[b, node_mapping]
            fused_raw_inputs = torch.cat([gathered_vis, text_features[b]], dim=-1)
            
            matched_out = self.matched_projection(fused_raw_inputs)
            unmatched_out = self.unmatched_projection(text_features[b])
            
            sample_realized = torch.where(is_matched_node.unsqueeze(-1), matched_out, unmatched_out)
            realized_nodes[b] = sample_realized * text_mask[b].unsqueeze(-1)
            
        if valid_batch_counts > 0:
            aux_alignment_loss = aux_alignment_loss / valid_batch_counts
            
        return realized_nodes, aux_alignment_loss