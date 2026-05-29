import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, global_mean_pool, global_max_pool
from task_verification.matching import GraphNodeRealizer

class TaskVerificationGNN(nn.Module):
    def __init__(self, visual_dim=768, text_dim=256, hidden_dim=256, dropout=0.4):
        """
        Substep 4 GNN Classifier: Flattens dense tensors into contiguous PyG graphs,
        executes message passing over DAG topologies, and predicts task anomalies.
        """
        super(TaskVerificationGNN, self).__init__()
        
        self.node_realizer = GraphNodeRealizer(
            visual_dim=visual_dim, text_dim=text_dim, joint_dim=hidden_dim, dropout=dropout
        )

        # Structural depth embedding for GNN message passing
        self.depth_embedding = nn.Embedding(50, hidden_dim, padding_idx=0)
        nn.init.normal_(self.depth_embedding.weight, std=0.01)
        
        # graph message passing layers
        self.gnn1 = GraphConv(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        
        self.gnn2 = GraphConv(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # graph-level global anomaly classification head
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, visual_features, text_features, visual_mask, text_mask, edge_indices, node_depths):
        """
        Args:
            visual_features: [B, Max_N, 768]
            text_features: [B, Max_M, 256]
            visual_mask: [B, Max_N]
            text_mask: [B, Max_M]
            edge_indices: List of length B containing [2, E_i] tensors
            node_depths: [B, Max_M] - structural depth of each node for positional encoding
        """
        batch_size = visual_features.size(0)
        device = visual_features.device
        
        # node feature realization -> [B, Max_M, 256] 
        realized_nodes, align_loss = self.node_realizer(visual_features, text_features, visual_mask, text_mask, node_depths)
        
        # graph batching (unrolling dense representations into flat tensors)
        x_list = []
        edge_list = []
        batch_assignment_list = []
        node_offset = 0
        
        for b in range(batch_size):
            num_real_nodes = int(text_mask[b].sum().item())
            
            # slice only valid, non-padded node states
            x_b = realized_nodes[b, :num_real_nodes]

            # injection of depth embedding for GNN message passing (structural PE)
            d_b = node_depths[b, :num_real_nodes].to(device)
            d_emb = self.depth_embedding(torch.clamp(d_b, max=49))
            x_b = x_b + d_emb 
            
            x_list.append(x_b)
            
            edge_idx = edge_indices[b].clone().to(device)
            if edge_idx.numel() > 0:
                edge_idx += node_offset
            edge_list.append(edge_idx)
            
            batch_b = torch.full((num_real_nodes,), fill_value=b, dtype=torch.long, device=device)
            batch_assignment_list.append(batch_b)
            
            node_offset += num_real_nodes
            
        # concatenate into contiguous sparse PyG graphs ready for message passing
        x_flat = torch.cat(x_list, dim=0)
        edge_index_flat = torch.cat(edge_list, dim=1) if len(edge_list) > 0 else torch.empty((2, 0), dtype=torch.long, device=device)
        batch_flat = torch.cat(batch_assignment_list, dim=0)
        
        # graph convolutional message passing block
        # first layer with ReLU and dropout
        h = self.gnn1(x_flat, edge_index_flat)
        h = self.ln1(h)
        h = F.relu(h)
        h = self.dropout(h)
        
        # 2nd layer with residual connection
        h_next = self.gnn2(h, edge_index_flat)
        h_next = self.ln2(h_next)
        h = F.relu(h_next + h) # residual injection
        h = self.dropout(h)
        
        # multi-Pooling graph readout (combines mean and max graph signals)
        pooled_mean = global_mean_pool(h, batch_flat, size=batch_size)
        pooled_max = global_max_pool(h, batch_flat, size=batch_size)
        graph_embedding = torch.cat([pooled_mean, pooled_max], dim=-1) # [B, hidden_dim * 2]
        
        # compute anomaly logits
        logits = self.classification_head(graph_embedding)
        return logits.squeeze(-1), align_loss