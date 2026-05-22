import argparse
import json
import os
import random
import numpy as np
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum
from torch_geometric.utils import softmax as pyg_softmax
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut


@dataclass
class GraphSample:
    video_id: str
    edge_index: torch.Tensor      # (2,E) long
    x_text: torch.Tensor          # (V,Dt) float
    x_vis_node: torch.Tensor      # (V,Dv) float (aligned step feature per node; 0 if unmatched)
    x_sim: torch.Tensor           # (V,1) float (match similarity for each node)
    y: int                        # 0/1


class GraphPTDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        pt_dir: Path,
        video_ids: List[str],
        labels: Dict[str, int],
    ):
        self.pt_dir = pt_dir
        self.video_ids = video_ids
        self.labels = labels

        missing = [vid for vid in video_ids if (pt_dir / f"{vid}.pt").exists() is False]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} pt files in {pt_dir}, e.g. {missing[:5]}")

        unlabeled = [vid for vid in video_ids if vid not in labels]
        if unlabeled:
            raise KeyError(f"Missing labels for {len(unlabeled)} videos, e.g. {unlabeled[:5]}")

    def __len__(self) -> int:
        return len(self.video_ids)

    def __getitem__(self, idx: int) -> GraphSample:
        vid = self.video_ids[idx]
        pack = _torch_load(self.pt_dir / f"{vid}.pt", map_location="cpu")

        if "edge_index" in pack:
            edge_index = pack["edge_index"].long()
        elif "edges" in pack and "step_ids" in pack:
            edge_index = build_edge_index(pack["edges"], pack["step_ids"])
        elif "task_graph" in pack and "step_ids" in pack:
            edge_index = build_edge_index(pack["task_graph"].get("edges", []), pack["step_ids"])
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        k_text = find_first_key(pack, ["text_embeddings", "x_text", "node_text_emb"])
        if k_text is None:
            raise KeyError(f"{vid}.pt missing text_embeddings/x_text/node_text_emb")
        x_text = pack[k_text].float()

        k_step = find_first_key(pack, ["visual_embeddings", "step_x", "step_emb"])
        if k_step is None:
            raise KeyError(f"{vid}.pt missing visual_embeddings/step_x/step_emb")
        step_x = pack[k_step].float()

        num_nodes = x_text.size(0)
        num_steps = step_x.size(0)
        node_to_step = build_node_to_step(pack, num_nodes, num_steps)

        x_vis_node = torch.zeros((num_nodes, step_x.size(1)), dtype=torch.float32)
        matched = node_to_step >= 0
        if matched.any():
            node_idx = torch.nonzero(matched, as_tuple=False).squeeze(-1)
            step_idx = node_to_step[node_idx]
            x_vis_node[node_idx] = step_x[step_idx]

        sims = pack.get("match_similarities", [])
        node_idx_list = pack.get("matched_node_indices", [])

        x_sim = torch.zeros(x_text.size(0), 1)
        for i, n_idx in enumerate(node_idx_list):
            if n_idx < x_sim.size(0):
                x_sim[n_idx] = sims[i]

        y = int(self.labels[vid])
        return GraphSample(video_id=vid, edge_index=edge_index, x_text=x_text, x_vis_node=x_vis_node, x_sim=x_sim, y=y)


# -------------------------
# Model Modules (RecipeVerifier Style)
# -------------------------
class NodeFusion(nn.Module):
    def __init__(self, dim_text: int, dim_vis: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.text_proj = nn.Linear(dim_text, hidden_dim)
        self.vis_proj = nn.Linear(dim_vis, hidden_dim)
        self.sim_proj = nn.Linear(2, hidden_dim) 
        
        # MLP profondo per dare molta più attenzione alla similarità e alle discrepanze
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, x_text: torch.Tensor, x_vis: torch.Tensor, x_sim: torch.Tensor) -> torch.Tensor:
        is_matched = (x_vis.abs().sum(dim=-1, keepdim=True) > 0).float()
        
        ht = self.text_proj(x_text)
        hv = self.vis_proj(x_vis)
        
        # Discrepanza esplicita (fondamentale per individuare errori di allineamento)
        discrepancy = torch.abs(ht - hv) * is_matched
        
        # Informazioni sulla qualità del match ungherese + maschera
        matching_info = torch.cat([x_sim, is_matched], dim=-1)
        hm = self.sim_proj(matching_info)

        # Concatenazione totale per permettere interazioni non lineari profonde
        h_cat = torch.cat([ht, hv, discrepancy, hm], dim=-1)
        return self.fusion_mlp(h_cat)


class DAGNNLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # Componenti dell'attenzione direzionale
        self.attn_w1 = nn.Linear(hidden_dim, 1, bias=False) 
        self.attn_w2 = nn.Linear(hidden_dim, 1, bias=False)
        # Cella GRU per aggiornare lo stato preservando la memoria cronologica
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x, edge_index, node_levels):
        src, dst = edge_index
        h_cur = x.clone()      # Stato dinamico aggiornato livello per livello
        h_prev_layer = x       # Caratteristiche statiche ereditate dal layer precedente
        
        max_level = node_levels.max().item() if node_levels.numel() > 0 else 0
        
        for level in range(max_level + 1):
            batch_nodes_mask = (node_levels == level)
            if not batch_nodes_mask.any():
                continue
            
            batch_nodes_idx = batch_nodes_mask.nonzero().view(-1)
            edge_mask = batch_nodes_mask[dst]
            
            if edge_mask.any():
                rel_src = src[edge_mask] 
                rel_dst = dst[edge_mask]
                
                query = self.attn_w1(h_prev_layer[rel_dst])  
                key = self.attn_w2(h_cur[rel_src]) 
                
                scores = query + key
                alpha = pyg_softmax(scores, rel_dst, num_nodes=x.size(0))
                
                weighted_messages = h_cur[rel_src] * alpha
                msgs_batch = scatter_sum(weighted_messages, rel_dst, dim=0, dim_size=x.size(0))[batch_nodes_idx]
            else:
                msgs_batch = torch.zeros((batch_nodes_idx.size(0), self.hidden_dim), device=x.device, dtype=x.dtype)

            h_batch_prev = h_prev_layer[batch_nodes_idx]
            h_batch_new = self.gru(h_batch_prev, msgs_batch)
            h_cur[batch_nodes_idx] = h_batch_new

        return h_cur


class AttentionalPooling(nn.Module):
    """Sostituisce AttentionalAggregation per garantire massima stabilità cross-platform"""
    def __init__(self, gate_nn):
        super().__init__()
        self.gate_nn = gate_nn

    def forward(self, x, batch, num_graphs):
        if x.numel() == 0:
            return torch.zeros((num_graphs, x.size(-1)), device=x.device, dtype=x.dtype)
        
        scores = self.gate_nn(x).squeeze(-1)
        alpha = pyg_softmax(scores, batch, num_nodes=num_graphs)
        
        weighted_x = x * alpha.unsqueeze(-1)
        return scatter_sum(weighted_x, batch, dim=0, dim_size=num_graphs)


class GraphClassifier(nn.Module):
    def __init__(
        self,
        dim_text: int,
        dim_vis: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.2,
        aggr: str = "mean", # Mantenuto per compatibilità con la chiamata principale
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.fusion = NodeFusion(dim_text, dim_vis, hidden_dim, dropout=dropout)

        # Norme e layer per i passaggi Forward e Backward
        self.norms_fwd = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.norms_bwd = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        self.fwd_layers = nn.ModuleList([DAGNNLayer(hidden_dim) for _ in range(num_layers)])
        self.bwd_layers = nn.ModuleList([DAGNNLayer(hidden_dim) for _ in range(num_layers)])

        cat_dim = (num_layers + 1) * hidden_dim
        readout_dim = 4 * cat_dim
        
        # Pooling attentivi dedicati separati per radici e foglie
        self.att_pool_fwd = AttentionalPooling(
            gate_nn=nn.Sequential(
                nn.Linear(cat_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        )
        
        self.att_pool_bwd = AttentionalPooling(
            gate_nn=nn.Sequential(
                nn.Linear(cat_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        )

        self.final_norm = nn.LayerNorm(readout_dim)

        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(readout_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_text: torch.Tensor, x_vis: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, x_sim: torch.Tensor = None) -> torch.Tensor:
        if x_sim is None:
            x_sim = torch.zeros((x_text.size(0), 1), device=x_text.device, dtype=x_text.dtype)
            
        # 1. Fusione multimodale con focus potenziato sulla similarità
        h0 = self.fusion(x_text, x_vis, x_sim)

        num_nodes = x_text.size(0)
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

        # 2. Calcolo dei livelli topologici dinamici
        topo_fwd = compute_topo_levels(edge_index, num_nodes)
        
        edge_index_bwd = torch.stack([edge_index[1], edge_index[0]], dim=0) if edge_index.numel() > 0 else torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
        topo_bwd = compute_topo_levels(edge_index_bwd, num_nodes)

        # FORWARD PASS: Flusso Radice -> Foglia con Skip connections
        h_fwd = h0
        outputs_fwd = [h0]
        for i, layer in enumerate(self.fwd_layers):
            h_new = layer(h_fwd, edge_index, topo_fwd)
            h_fwd = self.norms_fwd[i](h_fwd + h_new)
            outputs_fwd.append(h_fwd)

        # BACKWARD PASS: Flusso Foglia -> Radice rovesciato
        h_bwd = h0
        outputs_bwd = [h0]
        for i, layer in enumerate(self.bwd_layers):
            h_new = layer(h_bwd, edge_index_bwd, topo_bwd)
            h_bwd = self.norms_bwd[i](h_bwd + h_new)
            outputs_bwd.append(h_bwd)

        H_fwd_cat = torch.cat(outputs_fwd, dim=1)
        H_bwd_cat = torch.cat(outputs_bwd, dim=1)

        # 3. Readout Topologico mirato sugli Endpoint
        if edge_index.numel() > 0:
            out_degree = torch.bincount(edge_index[0], minlength=num_nodes)
            in_degree = torch.bincount(edge_index[1], minlength=num_nodes)
        else:
            out_degree = torch.zeros(num_nodes, dtype=torch.long, device=x_text.device)
            in_degree = torch.zeros(num_nodes, dtype=torch.long, device=x_text.device)
            
        target_mask = (out_degree == 0) # Nodi foglia (fine dei passaggi)
        source_mask = (in_degree == 0)  # Nodi radice (inizio dei passaggi)

        fwd_max = global_max_pool(H_fwd_cat[target_mask], batch[target_mask], num_graphs)
        fwd_att = self.att_pool_fwd(H_fwd_cat[target_mask], batch[target_mask], num_graphs)

        bwd_max = global_max_pool(H_bwd_cat[source_mask], batch[source_mask], num_graphs)
        bwd_att = self.att_pool_bwd(H_bwd_cat[source_mask], batch[source_mask], num_graphs)

        # Concatenazione globale delle scomposizioni strutturali
        graph_emb = torch.cat([fwd_max, fwd_att, bwd_max, bwd_att], dim=1)
        graph_emb = self.final_norm(graph_emb)

        logits = self.head(graph_emb).squeeze(-1)
        return logits
    

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.7, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean() if self.reduction == 'mean' else loss.sum()


def _torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return torch.load(*args, **kwargs)


def build_edge_index(edges: Any, step_ids: Any) -> torch.Tensor:
    if edges is None:
        return torch.empty((2, 0), dtype=torch.long)

    if isinstance(step_ids, dict):
        step_ids = list(step_ids.keys())
    step_id_to_index = {str(k): i for i, k in enumerate(step_ids)}

    src_idx = []
    dst_idx = []
    for edge in edges:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            continue
        src, dst = edge
        src_key = str(src)
        dst_key = str(dst)
        if src_key in step_id_to_index and dst_key in step_id_to_index:
            src_idx.append(step_id_to_index[src_key])
            dst_idx.append(step_id_to_index[dst_key])

    if len(src_idx) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor([src_idx, dst_idx], dtype=torch.long)


def build_node_to_step(pack: Dict[str, Any], num_nodes: int, num_steps: int) -> torch.Tensor:
    if "node_to_step" in pack:
        return pack["node_to_step"].long()

    node_to_step = torch.full((num_nodes,), -1, dtype=torch.long)
    if "matched_node_indices" in pack and "matched_visual_indices" in pack:
        vis_indices = pack["matched_visual_indices"]
        node_indices = pack["matched_node_indices"]
        if len(vis_indices) != len(node_indices):
            raise ValueError("matched_visual_indices and matched_node_indices lengths differ")
        for node_idx, step_idx in zip(node_indices, vis_indices):
            node_to_step[int(node_idx)] = int(step_idx)
    return node_to_step


def find_first_key(d: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in d:
            return k
    return None


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(-x))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def try_import_sklearn_auc():
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        return roc_auc_score, average_precision_score
    except ImportError:
        print("Warning: scikit-learn not installed. AUC metrics will not be available.")
        return None, None


def norm_subset(s: str) -> Optional[str]:
    s2 = str(s).strip().lower()
    if s2 in {"training", "train"}:
        return "train"
    if s2 in {"validation", "val", "valid"}:
        return "val"
    if s2 in {"test", "testing"}:
        return "test"
    return None


def load_recordings_combined(recordings_json: Path, label_rule: str = "any_step_error") -> Tuple[Dict[str, int], Dict[str, List[str]]]:
    with recordings_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "database" in data and isinstance(data["database"], dict):
        db = data["database"]
    elif isinstance(data, dict):
        db = data
    else:
        raise ValueError(f"Expected dict in {recordings_json}, got {type(data)}")

    split: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    labels: Dict[str, int] = {}

    for vid, item in db.items():
        if not isinstance(item, dict):
            continue

        subset = norm_subset(item.get("subset", ""))
        if subset is not None:
            split[subset].append(str(vid))

        anns = item.get("annotations", [])
        any_step_error = False
        if isinstance(anns, list):
            for a in anns:
                if isinstance(a, dict) and bool(a.get("has_error", False)) is True:
                    any_step_error = True
                    break

        if label_rule == "any_step_error":
            labels[str(vid)] = 0 if any_step_error else 1
        elif label_rule == "video_has_error":
            labels[str(vid)] = 0 if bool(item.get("has_error", False)) else 1
        else:
            raise ValueError(f"Unknown label_rule={label_rule}")

    for k in split:
        split[k] = sorted(split[k])

    if not labels:
        raise RuntimeError(f"No labels parsed from {recordings_json}")

    return labels, split


def collate_graph_samples(batch: List[GraphSample]) -> Dict[str, torch.Tensor]:
    x_texts, x_viss, x_sims, edge_indices, batch_vecs, ys = [], [], [], [], [], []
    node_offset = 0

    for gi, s in enumerate(batch):
        V = s.x_text.size(0)
        x_texts.append(s.x_text)
        x_viss.append(s.x_vis_node)
        x_sims.append(s.x_sim)
        ys.append(s.y)

        ei = s.edge_index
        if ei.numel() > 0:
            ei = ei + node_offset
        edge_indices.append(ei)

        batch_vecs.append(torch.full((V,), gi, dtype=torch.long))
        node_offset += V

    x_text = torch.cat(x_texts, dim=0)
    x_vis = torch.cat(x_viss, dim=0)
    x_sim = torch.cat(x_sims, dim=0)
    edge_index = torch.cat(edge_indices, dim=1) if len(edge_indices) else torch.empty((2, 0), dtype=torch.long)
    batch_vec = torch.cat(batch_vecs, dim=0)
    y = torch.tensor(ys, dtype=torch.float32)

    return {"x_text": x_text, "x_vis": x_vis, "x_sim": x_sim, "edge_index": edge_index, "batch": batch_vec, "y": y}


# ---------------------------------------------------------------------------
# UTILITY: Calcolo dinamico dell'ordinamento topologico (Gestione dell'Ordine)
# ---------------------------------------------------------------------------
def compute_topo_levels(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Calcola dinamicamente il livello topologico di ciascun nodo all'interno del DAG batched.
    Garantisce il corretto ordinamento sequenziale senza modificare il Dataloader.
    """
    in_degree = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
    if edge_index.numel() > 0:
        in_degree.scatter_add_(0, edge_index[1], torch.ones_like(edge_index[1]))
    
    levels = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
    rem_in_degree = in_degree.clone()
    curr_level = 0
    visited = torch.zeros(num_nodes, dtype=torch.bool, device=edge_index.device)
    
    zero_in = (rem_in_degree == 0) & (~visited)
    while zero_in.any():
        levels[zero_in] = curr_level
        visited[zero_in] = True
        
        if edge_index.numel() > 0:
            src, dst = edge_index
            from_zero_in = zero_in[src]
            if from_zero_in.any():
                dst_to_reduce = dst[from_zero_in]
                rem_in_degree.scatter_add_(0, dst_to_reduce, -torch.ones_like(dst_to_reduce))
        
        zero_in = (rem_in_degree == 0) & (~visited)
        curr_level += 1
        
    if not visited.all():
        levels[~visited] = curr_level # Fallback di sicurezza in caso di cicli imprevisti
        
    return levels


def global_mean_pool(x: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    dim = x.size(-1)
    out = torch.zeros(num_graphs, dim, device=x.device, dtype=x.dtype).index_add(0, batch, x)
    cnt = torch.zeros(num_graphs, device=x.device, dtype=x.dtype).index_add(
        0, batch, torch.ones(batch.size(0), device=x.device, dtype=x.dtype)
    )
    return out / cnt.clamp(min=1.0).unsqueeze(-1)


def global_max_pool(x: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    dim = x.size(-1)
    if x.numel() == 0:
        return torch.zeros((num_graphs, dim), device=x.device, dtype=x.dtype)
    idx = batch.unsqueeze(1).expand(-1, dim)
    out = torch.full((num_graphs, dim), float("-inf"), device=x.device, dtype=x.dtype)
    out = out.scatter_reduce(0, idx, x, reduce="amax", include_self=True)
    out = out.masked_fill(out == float("-inf"), 0.0)
    return out


# -------------------------
# Train / Eval
# -------------------------
@torch.no_grad()
def eval_epoch(model: nn.Module, loader, device: torch.device) -> Dict[str, float]:
    model.eval()
    all_logits, all_y = [], []

    for batch in loader:
        x_text = batch["x_text"].to(device)
        x_vis = batch["x_vis"].to(device)
        edge_index = batch["edge_index"].to(device)
        bvec = batch["batch"].to(device)
        x_sim = batch["x_sim"].to(device)
        y = batch["y"].to(device)

        logits = model(x_text, x_vis, edge_index, bvec, x_sim)
        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    y = torch.cat(all_y, dim=0)

    prob = sigmoid(logits)
    pred = (prob >= 0.5).long()
    y_int = y.long()

    tp = int(((pred == 1) & (y_int == 1)).sum())
    tn = int(((pred == 0) & (y_int == 0)).sum())
    fp = int(((pred == 1) & (y_int == 0)).sum())
    fn = int(((pred == 0) & (y_int == 1)).sum())

    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-12, prec + rec)

    roc_auc_score, average_precision_score = try_import_sklearn_auc()
    auc = float("nan")
    pr_auc = float("nan")
    if roc_auc_score is not None:
        try:
            auc = float(roc_auc_score(y.numpy(), prob.numpy()))
            pr_auc = float(average_precision_score(y.numpy(), prob.numpy()))
        except Exception:
            pass

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc, "pr_auc": pr_auc}


def train_one_epoch(model: nn.Module, loader, device: torch.device, optimizer, grad_clip: float,
                    pos_weight: torch.Tensor, label_smoothing: float = 0.1) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0

    for batch in loader:
        x_text = batch["x_text"].to(device)
        x_vis = batch["x_vis"].to(device)
        edge_index = batch["edge_index"].to(device)
        bvec = batch["batch"].to(device)
        x_sim = batch["x_sim"].to(device)
        y = batch["y"].to(device)

        logits = model(x_text, x_vis, edge_index, bvec, x_sim)

        focal_loss_fn = FocalLoss(alpha=0.5, gamma=2.0)
        loss = focal_loss_fn(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        total_n += bs

    return total_loss / max(1, total_n)


# -------------------------
# Cross-Validation Metrics
# -------------------------
def print_model_metrics(fold, oof_preds, oof_labels, oof_probs):
    """
    oof_preds: hard labels (0 or 1)
    oof_labels: ground truth labels
    oof_probs: raw probabilities for Class 1
    """
    print(f"\n--- RESULTS AT FOLD {fold} ---")

    np_true = np.asarray(oof_labels).astype(int)
    np_pred = np.asarray(oof_preds).astype(int)
    np_probs = np.asarray(oof_probs)

    total_count = len(np_true)
    correct_count = (np_true == np_pred).sum()

    acc = correct_count / total_count if total_count > 0 else 0.0

    f1 = f1_score(np_true, np_pred, average='macro')

    try:
        auc = roc_auc_score(np_true, np_probs)
    except ValueError:
        auc = 0.5  # Default if only one class is present in the fold

    if WANDB_AVAILABLE:
        try:
            wandb.log({
                "Model_accuracy": acc,
                "Macro_F1": f1,
                "AUC": auc
            })
        except Exception:
            pass

    print(f"PROCESSED GRAPHS: {total_count}")
    print(f"ACCURACY:  {acc:.4f}")
    print(f"MACRO F1:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    print(f"PRED DIST: {np.bincount(np_pred)}")
    print("----------------------------------------\n")


def get_probs(model: nn.Module, loader, device: torch.device):
    """Returns (all_probs, all_labels, all_logits, preds) arrays from the given loader."""
    model.eval()
    all_probs, all_labels, all_logits_list = [], [], []

    with torch.no_grad():
        for batch in loader:
            x_text = batch["x_text"].to(device)
            x_vis = batch["x_vis"].to(device)
            edge_index = batch["edge_index"].to(device)
            bvec = batch["batch"].to(device)
            x_sim = batch["x_sim"].to(device)
            y = batch["y"]

            logits = model(x_text, x_vis, edge_index, bvec, x_sim).view(-1)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.view(-1).numpy())
            all_logits_list.append(logits.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_logits_out = np.concatenate(all_logits_list)

    preds = (all_probs >= 0.5).astype(int)
    return all_probs, all_labels, all_logits_out, preds


def find_optimal_threshold(all_probs, all_labels):
    best_threshold = 0.5
    best_f1 = 0

    # Test 100 different threshold values from 0.01 to 0.99
    thresholds = np.linspace(0.01, 0.99, 100)

    for t in thresholds:
        # Apply current threshold
        preds = (all_probs >= t).astype(int)
        # Calculate Macro F1
        current_f1 = f1_score(all_labels, preds, average='macro')

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = t

    # Final check on accuracy with the best threshold
    best_preds = (all_probs >= best_threshold).astype(int)
    best_acc = accuracy_score(all_labels, best_preds)

    print(f"Optimal Threshold: {best_threshold:.4f}")
    print(f"Improved Macro F1: {best_f1:.4f}")
    print(f"Improved Accuracy: {best_acc:.4f}")

    return best_threshold


def compute_detailed_metrics(all_labels, all_preds):
    # 1. Macro-averaged metrics (treats both classes equally)
    precision_macro = precision_score(all_labels, all_preds, average='macro')
    recall_macro = recall_score(all_labels, all_preds, average='macro')

    # 2. Per-class metrics (0=Correct, 1=Error)
    # This is crucial for your "Discussion of Error Sensitivity"
    precision_per_class = precision_score(all_labels, all_preds, average=None)
    recall_per_class = recall_score(all_labels, all_preds, average=None)

    print(f"PRECISION (Macro): {precision_macro:.4f}")
    print(f"RECALL (Macro):    {recall_macro:.4f}")
    print("-" * 30)
    print(f"CLASS 'Correct' (0) - Precision: {precision_per_class[0]:.4f}, Recall: {recall_per_class[0]:.4f}")
    print(f"CLASS 'Error'   (1) - Precision: {precision_per_class[1]:.4f}, Recall: {recall_per_class[1]:.4f}")

    return precision_macro, recall_macro


# ---------------------------------------------------------------------------
# Cross-Validation Engine (LOGO / LOO)
# ---------------------------------------------------------------------------
def build_model_for_cv(dim_text: int, dim_vis: int, args, device: torch.device) -> GraphClassifier:
    """Instantiate and move a fresh GraphClassifier to device."""
    return GraphClassifier(
        dim_text=dim_text,
        dim_vis=dim_vis,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        aggr=args.aggr,
    ).to(device)


def run_cross_validation(dataset: GraphPTDataset, dim_text: int, dim_vis: int,
                         strategy: str, args, groups=None):
    """
    Unified engine for LOGO and LOO Cross-Validation.
    strategy: 'logo' or 'loo'

    Groups for LOGO are derived from the first part of each video_id before the
    underscore, so "1_1" and "1_4" both belong to group "1".
    """

    use_wandb = args.wandb and WANDB_AVAILABLE

    # Create a directory for per-fold metrics
    preds_dir = os.path.join(args.output_dir, f"{strategy}_folds_metrics")
    os.makedirs(preds_dir, exist_ok=True)

    if use_wandb:
        wandb.define_metric("epoch_in_fold")
        wandb.define_metric("train_loss_fold", step_metric="epoch_in_fold")
        wandb.define_metric("train_acc_fold", step_metric="epoch_in_fold")
        wandb.define_metric("folds_processed")
        wandb.define_metric("running_cv_accuracy", step_metric="folds_processed")

    splitter = LeaveOneGroupOut() if strategy == 'logo' else LeaveOneOut()

    all_oof_probs = []
    all_oof_labels = []
    all_oof_preds = []

    device = torch.device(args.device)
    indices = np.arange(len(dataset))

    for fold, (train_idx, val_idx) in enumerate(splitter.split(indices, groups=groups), 1):

        metric_path = os.path.join(preds_dir, f"fold_{fold}_metrics.pt")

        if os.path.exists(metric_path):
            saved_data = _torch_load(metric_path)

            all_oof_labels.append(saved_data['label'])
            all_oof_preds.append(saved_data['pred'])
            all_oof_probs.append(saved_data['prob'])

            oof_preds = np.concatenate(all_oof_preds)
            oof_labels = np.concatenate(all_oof_labels)
            oof_probs = np.concatenate(all_oof_probs)

            t_hist = saved_data.get('train_loss_history', [])
            if t_hist and use_wandb:
                for e_idx, t_l in enumerate(t_hist):
                    wandb.log({
                        "train_loss_fold": t_l,
                        "epoch_in_fold": e_idx,
                    })

            print(f"Fold {fold} found in cache. Replaying logs and skipping...")
            continue

        print(f"\nFold: {fold}")
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset   = torch.utils.data.Subset(dataset, val_idx)

        train_loader = torch.utils.data.DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_graph_samples,
            pin_memory=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_graph_samples,
            pin_memory=True,
        )

        model = build_model_for_cv(dim_text, dim_vis, args, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # Compute pos_weight from this fold's training subset
        fold_train_vids = [dataset.video_ids[i] for i in train_idx]
        num_pos = sum(1 for v in fold_train_vids if dataset.labels[v] == 1)
        num_neg = len(fold_train_vids) - num_pos
        pos_weight = torch.tensor([num_neg / max(1, num_pos)], device=device)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
        )

        fold_train_losses = []

        for ep in range(1, args.epochs + 1):
            tr_loss = train_one_epoch(
                model, train_loader, device, optimizer,
                args.grad_clip, pos_weight, label_smoothing=args.label_smoothing
            )
            scheduler.step(tr_loss)
            current_lr = optimizer.param_groups[0]['lr']

            fold_train_losses.append(tr_loss)

            if use_wandb:
                wandb.log({
                    "epoch_in_fold": ep,
                    "train_loss_fold": tr_loss,
                    "current_lr": current_lr,
                })

            print(f"Epoch {ep:03d} | Train loss: {tr_loss:.4f} | lr: {current_lr:.2e}")

        probs, labels, logits, preds = get_probs(model, val_loader, device)

        all_oof_probs.append(probs)
        all_oof_preds.append(preds)
        all_oof_labels.append(labels)

        fold_metrics = {
            'fold': fold,
            'label': labels,
            'pred': preds,
            'prob': probs,
            'logits': logits,
            'train_loss_history': fold_train_losses,
        }
        torch.save(fold_metrics, metric_path)
        print(f"--> [Checkpoint] Fold {fold} metrics saved.")

        if fold > 0 and fold % 5 == 0:
            tmp_preds  = np.concatenate(all_oof_preds)
            tmp_labels = np.concatenate(all_oof_labels)
            tmp_probs  = np.concatenate(all_oof_probs)
            print_model_metrics(fold, tmp_preds, tmp_labels, tmp_probs)
            compute_detailed_metrics(tmp_labels, tmp_preds)

    oof_preds  = np.concatenate(all_oof_preds)
    oof_labels = np.concatenate(all_oof_labels)
    oof_probs  = np.concatenate(all_oof_probs)

    print_model_metrics(fold, oof_preds, oof_labels, oof_probs)
    compute_detailed_metrics(oof_labels, oof_preds)

    print("\n--- SEARCHING FOR OPTIMAL THRESHOLD ---")
    best_t = find_optimal_threshold(oof_probs, oof_labels)

    final_preds = (oof_probs >= best_t).astype(int)

    print("\n--- GLOBAL PERFORMANCE (OPTIMIZED THRESHOLD) ---")
    print_model_metrics(f"FINAL_OPT_{best_t:.2f}", final_preds, oof_labels, oof_probs)
    compute_detailed_metrics(oof_labels, final_preds)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--graph_pt_dir", required=True, help="Substep3 output dir (each video_id.pt)")
    ap.add_argument("--recordings_json", required=True, help="CaptainCook4D recordings-combined.json")
    ap.add_argument("--label_rule", choices=["any_step_error", "video_has_error"], default="any_step_error")

    ap.add_argument("--output_dir", required=True, help="Save checkpoints/logs here")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--warmup_epochs", type=int, default=5)
    ap.add_argument("--early_stopping_patience", type=int, default=20)

    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--num_layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--aggr", choices=["sum", "mean"], default="mean")
    ap.add_argument("--label_smoothing", type=float, default=0.1)

    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--resume", default="", help="Path to checkpoint .pt")

    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", default="gnn-error-detection")
    ap.add_argument("--wandb_run_name", default="")
    ap.add_argument("--wandb_entity", default="")
    
    ap.add_argument("--strategy", choices=["train_all", "logo", "loo"], default="train_all", help=("Validation strategy: ""'train_all' uses the predefined train/val/test split from recordings_json; ""'logo' runs LeaveOneGroupOut CV grouped by recipe-id (part before '_' in video_id); ""'loo' runs LeaveOneOut CV over all usable samples."),)

    args = ap.parse_args()

    set_seed(args.seed)

    use_wandb = args.wandb and WANDB_AVAILABLE and not args.eval_only
    if args.wandb and not WANDB_AVAILABLE:
        print("Warning: wandb not installed. Run 'pip install wandb'. Proceeding without logging.")

    pt_dir = Path(args.graph_pt_dir)
    labels, split = load_recordings_combined(Path(args.recordings_json), label_rule=args.label_rule)

    pt_vids = sorted(p.stem for p in pt_dir.glob("*.pt"))
    labeled_vids = set(labels.keys())
    usable = sorted(v for v in pt_vids if v in labeled_vids)
    if not usable:
        raise RuntimeError(f"No usable samples: check {pt_dir} and recordings_json labels")

    # Determine dim_text and dim_vis from the first available sample
    sample0 = _torch_load(pt_dir / f"{usable[0]}.pt", map_location="cpu")
    k_text = find_first_key(sample0, ["text_embeddings", "x_text", "node_text_emb"])
    k_step = find_first_key(sample0, ["visual_embeddings", "step_x", "step_emb"])
    if k_text is None or k_step is None:
        raise KeyError(f"pt sample missing required keys. Found keys: {list(sample0.keys())}")
    dim_text = int(sample0[k_text].shape[1])
    dim_vis  = int(sample0[k_step].shape[1])

    # ------------------------------------------------------------------
    # LOGO / LOO branch
    # ------------------------------------------------------------------
    if args.strategy in ("logo", "loo"):
        # Build full dataset over all usable video ids
        full_dataset = GraphPTDataset(pt_dir, usable, labels)

        # Build groups array: recipe-id = part of video_id before the first '_'
        # e.g. "1_1" -> "1", "2_3" -> "2"
        groups = np.array([vid.split("_")[0] for vid in usable])

        if use_wandb:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or f"{args.strategy}_run",
                entity=args.wandb_entity or None,
                config={
                    "strategy":       args.strategy,
                    "seed":           args.seed,
                    "epochs":         args.epochs,
                    "batch_size":     args.batch_size,
                    "lr":             args.lr,
                    "weight_decay":   args.weight_decay,
                    "grad_clip":      args.grad_clip,
                    "hidden_dim":     args.hidden_dim,
                    "num_layers":     args.num_layers,
                    "dropout":        args.dropout,
                    "aggr":           args.aggr,
                    "label_rule":     args.label_rule,
                    "label_smoothing": args.label_smoothing,
                    "dim_text":       dim_text,
                    "dim_vis":        dim_vis,
                    "total_samples":  len(usable),
                },
            )

        run_cross_validation(
            dataset=full_dataset,
            dim_text=dim_text,
            dim_vis=dim_vis,
            strategy=args.strategy,
            args=args,
            groups=groups if args.strategy == "logo" else None,
        )

        if use_wandb:
            wandb.finish()
        return

    # ------------------------------------------------------------------
    # TRAIN_ALL branch (original train/val/test split logic)
    # ------------------------------------------------------------------
    train_vids = [v for v in split.get("train", []) if v in usable]
    val_vids   = [v for v in split.get("val",   []) if v in usable]
    test_vids  = [v for v in split.get("test",  []) if v in usable]

    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or None,
            entity=args.wandb_entity or None,
            config={
                "strategy":       args.strategy,
                "seed":           args.seed,
                "epochs":         args.epochs,
                "batch_size":     args.batch_size,
                "lr":             args.lr,
                "weight_decay":   args.weight_decay,
                "grad_clip":      args.grad_clip,
                "hidden_dim":     args.hidden_dim,
                "num_layers":     args.num_layers,
                "dropout":        args.dropout,
                "aggr":           args.aggr,
                "label_rule":     args.label_rule,
                "label_smoothing": args.label_smoothing,
            },
        )
        wandb.config.update({
            "train_size": len(train_vids),
            "val_size":   len(val_vids),
            "test_size":  len(test_vids),
        }, allow_val_change=True)

    train_ds = GraphPTDataset(pt_dir, train_vids, labels)
    val_ds   = GraphPTDataset(pt_dir, val_vids,   labels) if val_vids  else None
    test_ds  = GraphPTDataset(pt_dir, test_vids,  labels) if test_vids else None

    g = torch.Generator()
    g.manual_seed(args.seed)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_graph_samples,
        pin_memory=True,
        generator=g,
        worker_init_fn=seed_worker,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_graph_samples,
        pin_memory=True,
    ) if val_ds else None
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_graph_samples,
        pin_memory=True,
    ) if test_ds else None

    device = torch.device(args.device)
    model = GraphClassifier(
        dim_text=dim_text,
        dim_vis=dim_vis,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        aggr=args.aggr,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(epoch: int) -> float:
        if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
            return float(epoch + 1) / float(args.warmup_epochs)
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1.0 + __import__('math').cos(__import__('math').pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if use_wandb:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.config.update({
            "trainable_params": total_params,
            "dim_text": dim_text,
            "dim_vis": dim_vis,
            "warmup_epochs": args.warmup_epochs,
            "early_stopping_patience": args.early_stopping_patience,
        }, allow_val_change=True)
        wandb.watch(model, log="gradients", log_freq=50)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_best = out_dir / "best.pt"
    ckpt_last = out_dir / "last.pt"

    if args.resume:
        ckpt = _torch_load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt and not args.eval_only:
            optimizer.load_state_dict(ckpt["optimizer"])
        print(f"[RESUME] loaded {args.resume}")

    if args.eval_only:
        if val_loader:
            print("[VAL]", eval_epoch(model, val_loader, device))
        if test_loader:
            print("[TEST]", eval_epoch(model, test_loader, device))
        return

    best_key = "auc"
    best_score = -1e9
    epochs_without_improvement = 0

    num_pos = sum(1 for v in train_vids if labels[v] == 1)
    num_neg = len(train_vids) - num_pos
    pos_weight = torch.tensor([num_neg / max(1, num_pos)], device=device)

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, device, optimizer, args.grad_clip, pos_weight, label_smoothing=args.label_smoothing)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        val_metrics = None
        if val_loader:
            val_metrics = eval_epoch(model, val_loader, device)

        if val_metrics:
            score = val_metrics.get(best_key, float("nan"))
            if score != score:
                score = val_metrics.get("f1", 0.0)
        else:
            score = -tr_loss

        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}, ckpt_last)

        if score > best_score:
            best_score = score
            epochs_without_improvement = 0
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "best_score": best_score},
                ckpt_best
            )
        else:
            epochs_without_improvement += 1

        if val_metrics:
            print(f"Epoch {epoch:03d} | loss={tr_loss:.4f} | lr={current_lr:.2e} | val={val_metrics} | best_score={best_score:.4f} | no_improve={epochs_without_improvement}")
        else:
            print(f"Epoch {epoch:03d} | loss={tr_loss:.4f} | lr={current_lr:.2e} | best_score={best_score:.4f}")

        if use_wandb:
            log_dict = {"epoch": epoch, "train/loss": tr_loss, "train/lr": current_lr, "train/best_score": best_score}
            if val_metrics:
                log_dict.update({f"val/{k}": v for k, v in val_metrics.items()})
            wandb.log(log_dict)

        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(f"[EARLY STOP] No improvement for {args.early_stopping_patience} epochs. Stopping at epoch {epoch}.")
            if use_wandb:
                wandb.summary["early_stop_epoch"] = epoch
            break

    if ckpt_best.exists():
        ckpt = _torch_load(ckpt_best, map_location="cpu")
        model.load_state_dict(ckpt["model"])

    if test_loader:
        test_metrics = eval_epoch(model, test_loader, device)
        print("[FINAL TEST]", test_metrics)
        if use_wandb:
            wandb.log({f"test/{k}": v for k, v in test_metrics.items()})
            wandb.summary.update({f"test/{k}": v for k, v in test_metrics.items()})

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()