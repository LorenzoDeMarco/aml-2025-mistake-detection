import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class GraphNodeRealizer(nn.Module):
    def __init__(self, visual_dim=768, text_dim=256, joint_dim=256, dropout=0.4):
        """
        Video context token alignment with Hungarian Graph matching.
        Unified projection and absolute index mapping implementation.
        """
        super(GraphNodeRealizer, self).__init__()
        
        # temporal contractive convolution (Stride 4 with explicit protective padding)
        self.temporal_conv = nn.Conv1d(
            in_channels=visual_dim,
            out_channels=visual_dim,
            kernel_size=4,
            stride=4,
            padding=1
        )
        
        # metric space mapping projections
        self.sim_visual_proj = nn.Linear(visual_dim, joint_dim)
        self.sim_text_proj = nn.Linear(text_dim, joint_dim)
        
        # fusion head 
        self.unified_fusion = nn.Sequential(
            nn.Linear(visual_dim + text_dim, joint_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(joint_dim * 2, joint_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(joint_dim)
        )
        
        # learnable missing visual Token
        self.missing_visual_embedding = nn.Parameter(torch.zeros(1, visual_dim))
        nn.init.normal_(self.missing_visual_embedding, std=0.02)
        
        self.temperature = 0.07

    def forward(self, visual_features, text_features, visual_mask, text_mask):
        batch_size, max_m, text_dim = text_features.shape
        device = visual_features.device
        
        #dynamic temporal pooling step
        vis_transposed = visual_features.transpose(1, 2)
        compressed_vis = self.temporal_conv(vis_transposed).transpose(1, 2)
        compressed_mask = visual_mask[:, ::4][:, :compressed_vis.size(1)]
        comp_n = compressed_vis.size(1)
        
        # project modalities into common metric space
        proj_v = self.sim_visual_proj(compressed_vis)
        proj_t = self.sim_text_proj(text_features)
        
        realized_nodes = torch.zeros(batch_size, max_m, self.sim_text_proj.out_features, device=device)
        extended_visual = torch.cat([compressed_vis, self.missing_visual_embedding.expand(batch_size, 1, -1)], dim=1)
        
        aux_alignment_loss = torch.tensor(0.0, device=device)
        valid_batch_counts = 0
        
        for b in range(batch_size):
            num_vis = int(compressed_mask[b].sum().item())
            num_text = int(text_mask[b].sum().item())
            
            #default lookup arrays tracking mapping positions 
            node_mapping = torch.full((max_m,), fill_value=comp_n, dtype=torch.long, device=device)
            
            if num_vis > 0 and num_text > 0:
                v_norm = F.normalize(proj_v[b, :num_vis], p=2, dim=-1)
                t_norm = F.normalize(proj_t[b, :num_text], p=2, dim=-1)
                
                similarity_matrix = torch.mm(v_norm, t_norm.t()) # [num_vis, num_text]
                
                # Hungarian Matching execution
                cost_matrix = 1.0 - similarity_matrix.detach().cpu().numpy()
                v_idx, t_idx = linear_sum_assignment(cost_matrix)
                
                if len(v_idx) > 0:
                    v_idx_t = torch.tensor(v_idx, device=device)
                    t_idx_t = torch.tensor(t_idx, device=device)
                    
                    # --- INFONCE HUNGARIAN CONTRASTIVE LOSS ---
                    logits = similarity_matrix / self.temperature
                    matched_logits = logits.t()[t_idx_t] # [num_matched, num_vis]
                    
                    contrastive_loss = F.cross_entropy(matched_logits, v_idx_t)
                    aux_alignment_loss += contrastive_loss
                    valid_batch_counts += 1
                    
                    # Cosine Semantic Thresholding (0.20)
                    matched_sims = similarity_matrix[v_idx_t, t_idx_t]
                    
                    for i in range(len(v_idx)):
                        if matched_sims[i] >= 0.20:
                            # Map the local index safely into absolute token coordinate positions
                            node_mapping[t_idx[i]] = v_idx[i]
                            
            gathered_vis = extended_visual[b, node_mapping]
            
            fused_inputs = torch.cat([gathered_vis, text_features[b]], dim=-1)
            sample_realized = self.unified_fusion(fused_inputs)
            
            realized_nodes[b] = sample_realized * text_mask[b].unsqueeze(-1)
            
        if valid_batch_counts > 0:
            aux_alignment_loss = aux_alignment_loss / valid_batch_counts
            
        return realized_nodes, aux_alignment_loss