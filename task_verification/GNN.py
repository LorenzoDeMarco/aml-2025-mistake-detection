import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, global_mean_pool, global_max_pool
from task_verification.matching import GraphNodeRealizer

class TaskVerificationGNN(nn.Module):
    def __init__(self, visual_dim=768, text_dim=256, hidden_dim=256, dropout=0.4):
        super(TaskVerificationGNN, self).__init__()
        
        self.node_realizer = GraphNodeRealizer(
            visual_dim=visual_dim, text_dim=text_dim, joint_dim=hidden_dim, dropout=dropout
        )
        
        self.gnn1 = GraphConv(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        
        self.gnn2 = GraphConv(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        self.gnn3 = GraphConv(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, visual_features, text_features, visual_mask, text_mask, edge_indices):
        batch_size = visual_features.size(0)
        device = visual_features.device
        
        realized_nodes, align_loss = self.node_realizer(visual_features, text_features, visual_mask, text_mask)
        
        x_list = []
        edge_list = []
        batch_assignment_list = []
        node_offset = 0
        
        for b in range(batch_size):
            num_real_nodes = int(text_mask[b].sum().item())
            
            x_b = realized_nodes[b, :num_real_nodes]
            x_list.append(x_b)
            
            edges_b = edge_indices[b].to(device)
            if edges_b.numel() > 0:
                edge_list.append(edges_b + node_offset)
                
            batch_b = torch.full((num_real_nodes,), fill_value=b, dtype=torch.long, device=device)
            batch_assignment_list.append(batch_b)
            
            node_offset += num_real_nodes
            
        x_flat = torch.cat(x_list, dim=0)
        edge_index_flat = torch.cat(edge_list, dim=1) if len(edge_list) > 0 else torch.empty((2, 0), dtype=torch.long, device=device)
        batch_flat = torch.cat(batch_assignment_list, dim=0)
        
        h = self.gnn1(x_flat, edge_index_flat)
        h = self.ln1(h)
        h = F.relu(h)
        h = self.dropout(h)
        
        h_next = self.gnn2(h, edge_index_flat)
        h_next = self.ln2(h_next)
        h = F.relu(h_next + h)
        h = self.dropout(h)
        
        h_deep = self.gnn3(h, edge_index_flat)
        h_deep = self.ln3(h_deep)
        h = F.relu(h_deep + h)
        h = self.dropout(h)
        
        pooled_mean = global_mean_pool(h, batch_flat, size=batch_size)
        pooled_max = global_max_pool(h, batch_flat, size=batch_size)
        graph_embedding = torch.cat([pooled_mean, pooled_max], dim=-1)
        
        logits = self.classification_head(graph_embedding)
        return logits.squeeze(-1), align_loss