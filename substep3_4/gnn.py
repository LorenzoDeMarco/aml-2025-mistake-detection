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
from turtle import ht
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GraphSample:
    video_id: str
    edge_index: torch.Tensor      # (2,E) long
    x_text: torch.Tensor          # (V,Dt) float
    x_vis_node: torch.Tensor      # (V,Dv) float (aligned step feature per node; 0 if unmatched)
    x_sim: torch.Tensor           # (V,1) float (match similarity for each node)
    y: int                        # 0/1




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


class GraphPTDataset(torch.utils.data.Dataset):
    """
    Reads Substep3 output .pt:
      required keys:
        - x_text OR node_text_emb OR text_embeddings
        - step_x OR step_emb OR visual_embeddings
        - node_to_step OR matched_node_indices + matched_visual_indices
      optional:
        - edge_index OR edges + step_ids OR task_graph
      label from labels dict
    """
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
        pack = torch.load(self.pt_dir / f"{vid}.pt", map_location="cpu")

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

        # Estrai match_similarities da taskGraphEncoding e crea x_sim per ogni nodo
        sims = pack.get("match_similarities", [])
        node_idx_list = pack.get("matched_node_indices", [])

        x_sim = torch.zeros(x_text.size(0), 1)
        for i, n_idx in enumerate(node_idx_list):
            if n_idx < x_sim.size(0):
                x_sim[n_idx] = sims[i]

        y = int(self.labels[vid])
        return GraphSample(video_id=vid, edge_index=edge_index, x_text=x_text, x_vis_node=x_vis_node, x_sim=x_sim, y=y)

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
    # Forza operazioni CUDA deterministiche (es. index_add_, scatter)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Lancia eccezione se un'op non ha implementazione deterministica
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
    
def norm_subset(s: str) -> Optional[str]: #give the substep value and return the normalized subset name
        s2 = str(s).strip().lower()
        if s2 in {"training", "train"}:
            return "train"
        if s2 in {"validation", "val", "valid"}:
            return "val"
        if s2 in {"test", "testing"}:
            return "test"
        return None
    
def load_recordings_combined(recordings_json: Path, label_rule: str = "any_step_error") -> Tuple[Dict[str, int], Dict[str, List[str]]]:
    """
    recordings-combined.json structure:
      {
        "version": "...",
        "database": {
            "1_19": {"subset":"Training","annotations":[{"has_error":...}, ...], ...},
            ...
        }
      }
    """
    with recordings_json.open("r", encoding="utf-8") as f:
        data = json.load(f)


    #  real videos are in data["database"]
    if isinstance(data, dict) and "database" in data and isinstance(data["database"], dict):
        db = data["database"] #field in the json file that contains the video data
    elif isinstance(data, dict):
        # fallback: assume whole dict is db
        db = data
    else:
        raise ValueError(f"Expected dict in {recordings_json}, got {type(data)}")


    split: Dict[str, List[str]] = {"train": [], "val": [], "test": []} #inizialization of the dict
    labels: Dict[str, int] = {}


    for vid, item in db.items(): #vid= video_id, item= dict with video info
        if not isinstance(item, dict):
            continue


        # split（ Training/Validation/Test）
        subset = norm_subset(item.get("subset", "")) # pass the subset value
        if subset is not None:
            split[subset].append(str(vid)) # add the video_id to the corresponding subset list


        # label： default use "any step has_error==True => error(0) else correct(1)"
        anns = item.get("annotations", []) # sequence of step with has_error field
        any_step_error = False
        if isinstance(anns, list):
            for a in anns:
                if isinstance(a, dict) and bool(a.get("has_error", False)) is True: # if no step has error, has_error is False or missing, else True
                    any_step_error = True
                    break


        if label_rule == "any_step_error":
            labels[str(vid)] = 0 if any_step_error else 1
        elif label_rule == "video_has_error":
            labels[str(vid)] = 0 if bool(item.get("has_error", False)) else 1
        else:
            raise ValueError(f"Unknown label_rule={label_rule}")


    # keep deterministic order
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




# -------------------------
# Model
# -------------------------
class NodeFusion(nn.Module):
    def __init__(self, dim_text: int, dim_vis: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.text_proj = nn.Linear(dim_text, hidden_dim)
        self.vis_proj = nn.Linear(dim_vis, hidden_dim)
        # Proiettiamo la similarità dell'Hungarian (1D) e la maschera (1D)
        self.sim_proj = nn.Linear(2, hidden_dim) 
        
        self.ln = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x_text: torch.Tensor, x_vis: torch.Tensor, x_sim: torch.Tensor) -> torch.Tensor:
        # Maschera: 1 se c'è un match, 0 altrimenti
        is_matched = (x_vis.abs().sum(dim=-1, keepdim=True) > 0).float()
        
        ht = self.text_proj(x_text)
        hv = self.vis_proj(x_vis)
        
        # 1. DIFFERENZA ESPLICITA: Se testo e video non coincidono, questo vettore esplode
        # È il segnale principale per il mistake detection
        discrepancy = torch.abs(ht - hv) * is_matched
        
        # 2. INFO MATCHING: Quanto CLIP era convinto del match?
        # x_sim deve contenere il valore numerico (es. 0.28) dall'Hungarian
        matching_info = torch.cat([x_sim, is_matched], dim=-1)
        hm = self.sim_proj(matching_info)

        # Fusione: Testo + Video + Errore + Qualità del Match
        h = ht + hv + discrepancy + hm
        return self.ln(F.gelu(h)) # GELU spesso aiuta più di ReLU in questi casi




class DiGraphConv(nn.Module):
    """
    Simple directed message passing:
      agg_v = sum/mean_{(u->v)} W_msg x_u
      x'_v = act( W_self x_v + agg_v )
    """
    def __init__(self, dim: int, dropout: float = 0.1, aggr: str = "mean"):
        super().__init__()
        assert aggr in {"sum", "mean"}
        self.aggr = aggr
        self.msg = nn.Linear(dim, dim, bias=False)
        self.self_lin = nn.Linear(dim, dim, bias=True)
        self.ln = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)


    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            out = self.self_lin(x)
            out = F.relu(out)
            return self.ln(self.drop(out))

        src, dst = edge_index[0], edge_index[1]
        msg = self.msg(x[src])

        # torch.zeros crea un tensore fresco senza storia nel grafo —
        # index_add_ è sicuro qui perché non modifica tensori che
        # partecipano al backward (x non viene mai toccato in-place)
        agg = torch.zeros(x.size(0), x.size(1), device=x.device, dtype=x.dtype)
        agg = agg.index_add(0, dst, msg)   # out-of-place: restituisce nuovo tensore

        if self.aggr == "mean":
            deg = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
            deg = deg.index_add(0, dst, torch.ones(dst.size(0), device=x.device, dtype=x.dtype))
            agg = agg / deg.clamp(min=1.0).unsqueeze(-1)

        out = self.self_lin(x) + agg
        out = F.relu(out)
        out = self.ln(out)
        return self.drop(out)




def global_mean_pool(x: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    dim = x.size(-1)
    out = torch.zeros(num_graphs, dim, device=x.device, dtype=x.dtype).index_add(0, batch, x)
    cnt = torch.zeros(num_graphs, device=x.device, dtype=x.dtype).index_add(
        0, batch, torch.ones(batch.size(0), device=x.device, dtype=x.dtype)
    )
    return out / cnt.clamp(min=1.0).unsqueeze(-1)


def global_max_pool(x: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """
    Max pooling over nodes per graph tramite scatter (out-of-place, differenziabile).
    Focalizza il classificatore sui nodi più attivi/anomali
    invece di diluire il segnale con la media.
    Usa torch.scatter_reduce per essere completamente differenziabile e
    senza alcuna operazione in-place sul grafo computazionale.
    """
    dim = x.size(-1)
    # Espandi batch index a (N, dim) per scatter_reduce
    idx = batch.unsqueeze(1).expand(-1, dim)   # (N, dim)
    # scatter_reduce con 'amax': completamente out-of-place e differenziabile
    out = torch.full((num_graphs, dim), float("-inf"), device=x.device, dtype=x.dtype)
    out = out.scatter_reduce(0, idx, x, reduce="amax", include_self=True)
    # Fallback: grafi senza nodi (edge case) -> zero invece di -inf
    out = out.masked_fill(out == float("-inf"), 0.0)
    return out




class GraphClassifier(nn.Module):
    def __init__(
        self,
        dim_text: int,
        dim_vis: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.2,
        aggr: str = "mean",
    ):
        super().__init__()
        self.fusion = NodeFusion(dim_text, dim_vis, hidden_dim, dropout=dropout)
        self.layers = nn.ModuleList([DiGraphConv(hidden_dim, dropout=dropout, aggr=aggr) for _ in range(num_layers)])

        # MODIFICA: Il primo layer lineare ora riceve hidden_dim * 2
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform per tutti i Linear: evita varianza alta con hidden_dim=256
        che con l'init default di PyTorch (kaiming_uniform) può destabilizzare il training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, x_text: torch.Tensor, x_vis: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, x_sim: torch.Tensor = None) -> torch.Tensor:
        if x_sim is None:
            x_sim = torch.zeros((x_text.size(0), 1), device=x_text.device, dtype=x_text.dtype)
        h = self.fusion(x_text, x_vis, x_sim)
        for layer in self.layers:
            h = h + layer(h, edge_index)  # residual
        
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        
        # MODIFICA: Calcola sia Mean che Max Pool
        g_mean = global_mean_pool(h, batch, num_graphs)
        g_max = global_max_pool(h, batch, num_graphs)
        
        # Concatena i due vettori lungo la dimensione delle feature
        g = torch.cat([g_mean, g_max], dim=-1) # Dimensione risultante: (num_graphs, hidden_dim * 2)
        
        logits = self.head(g).squeeze(-1)
        return logits
    

class FocalLoss(nn.Module): #Aggiunta per migliorare
    def __init__(self, alpha=0.7, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha # Bilanciamento classi (0.25 favorisce la minoranza se gamma > 0)
        self.gamma = gamma # Fattore di "focus" sugli esempi difficili
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

        # Label smoothing: trasforma target 0->eps, 1->(1-eps).
        # Evita che il modello si "specializzi" troppo su target duri 0/1,
        # riducendo la tendenza a collassare su previsioni degeneri (recall=1 o recall=0).
        if label_smoothing > 0.0:
            y_smooth = y * (1.0 - label_smoothing) + 0.5 * label_smoothing
        else:
            y_smooth = y

        focal_loss_fn = FocalLoss(alpha=0.5, gamma=2.0) # Prova alpha 0.5 per bilanciare equamente
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




def main():
    ap = argparse.ArgumentParser()


    ap.add_argument("--graph_pt_dir", required=True, help="Substep3 output dir (each video_id.pt)")
    ap.add_argument("--recordings_json", required=True, help="CaptainCook4D recordings-combined.json")
    ap.add_argument("--label_rule", choices=["any_step_error", "video_has_error"], default="any_step_error",
                    help="How to build video-level label: default uses annotations[*].has_error.")


    ap.add_argument("--output_dir", required=True, help="Save checkpoints/logs here")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


    # if subset missing, fallback random split
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=123)


    # training
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="Peak LR dopo warmup. Ridotto da 3e-4 per stabilizzare training su dataset piccoli.")
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--warmup_epochs", type=int, default=5,
                    help="Epoch di linear warmup prima del cosine decay.")
    ap.add_argument("--early_stopping_patience", type=int, default=20,
                    help="Stop se val AUC non migliora per N epoch. 0 = disabilitato.")


    # model
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--num_layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1,
                    help="Dropout ridotto da 0.2: con dataset piccoli dropout alto impedisce convergenza.")
    ap.add_argument("--aggr", choices=["sum", "mean"], default="mean")
    ap.add_argument("--label_smoothing", type=float, default=0.1, # aggiunto label smoothing per stabilizzare training e ridurre collasso su pred degeneri
                    help="Label smoothing su BCE: target 0->eps, 1->(1-eps). Riduce collasso su pred degeneri.")


    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--resume", default="", help="Path to checkpoint .pt")

    # wandb
    ap.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    ap.add_argument("--wandb_project", default="gnn-error-detection", help="W&B project name")
    ap.add_argument("--wandb_run_name", default="", help="W&B run name (auto if empty)")
    ap.add_argument("--wandb_entity", default="", help="W&B entity/team (optional)")


    args = ap.parse_args()


    set_seed(args.seed)

    # ── W&B init ──────────────────────────────────────────────────────────
    use_wandb = args.wandb and WANDB_AVAILABLE and not args.eval_only
    if args.wandb and not WANDB_AVAILABLE:
        print("Warning: wandb not installed. Run 'pip install wandb'. Proceeding without logging.")
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or None,
            entity=args.wandb_entity or None,
            config={
                "seed":         args.seed,
                "epochs":       args.epochs,
                "batch_size":   args.batch_size,
                "lr":           args.lr,
                "weight_decay": args.weight_decay,
                "grad_clip":    args.grad_clip,
                "hidden_dim":   args.hidden_dim,
                "num_layers":   args.num_layers,
                "dropout":      args.dropout,
                "aggr":         args.aggr,
                "label_rule":       args.label_rule,
                "label_smoothing":  args.label_smoothing,
            },
        )
    # ──────────────────────────────────────────────────────────────────────

    pt_dir = Path(args.graph_pt_dir)

    labels, split = load_recordings_combined(Path(args.recordings_json), label_rule=args.label_rule)

    # only keep vids that have .pt and label
    # Usa sorted() su entrambi prima dell'intersezione per garantire
    # ordine deterministico indipendente dall'hash interno dei set
    pt_vids = sorted(p.stem for p in pt_dir.glob("*.pt"))
    labeled_vids = set(labels.keys())
    usable = sorted(v for v in pt_vids if v in labeled_vids)
    if not usable:
        raise RuntimeError(f"No usable samples: check {pt_dir} and recordings_json labels")


    # build split from subset (preferred)
    train_vids = [v for v in split.get("train", []) if v in usable]
    val_vids = [v for v in split.get("val", []) if v in usable]
    test_vids = [v for v in split.get("test", []) if v in usable]

    if use_wandb:
        wandb.config.update({
            "train_size": len(train_vids),
            "val_size":   len(val_vids),
            "test_size":  len(test_vids),
        }, allow_val_change=True)

    train_ds = GraphPTDataset(pt_dir, train_vids, labels)
    val_ds = GraphPTDataset(pt_dir, val_vids, labels) if val_vids else None
    test_ds = GraphPTDataset(pt_dir, test_vids, labels) if test_vids else None


    # Generator con seed fisso: garantisce shuffle riproducibile
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


    # infer dims from one sample
    sample0 = torch.load(pt_dir / f"{train_vids[0]}.pt", map_location="cpu")
    k_text = find_first_key(sample0, ["text_embeddings", "x_text", "node_text_emb"])
    k_step = find_first_key(sample0, ["visual_embeddings", "step_x", "step_emb"])
    if k_text is None or k_step is None:
        raise KeyError(f"pt sample missing required keys. Found keys: {list(sample0.keys())}")


    dim_text = int(sample0[k_text].shape[1])
    dim_vis = int(sample0[k_step].shape[1])


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

    # Cosine annealing con linear warmup:
    # - warmup evita i grandi gradienti iniziali che causano collasso su pred degeneri
    # - cosine decay riduce il lr gradualmente invece di tenerlo fisso fino all'overfitting
    def lr_lambda(epoch: int) -> float:
        if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
            return float(epoch + 1) / float(args.warmup_epochs)  # linear warmup
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1.0 + __import__('math').cos(__import__('math').pi * progress))  # cosine decay

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


    # resume / eval-only
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
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


    best_key = "auc"  # prefer AUC if available; fallback to f1
    best_score = -1e9
    epochs_without_improvement = 0  # early stopping counter

    num_pos = sum(1 for v in train_vids if labels[v] == 1)
    num_neg = len(train_vids) - num_pos
    pos_weight = torch.tensor([num_neg / max(1, num_pos)], device=device)

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, device, optimizer, args.grad_clip, pos_weight, label_smoothing=args.label_smoothing)
        scheduler.step()  # aggiorna lr dopo ogni epoch
        current_lr = scheduler.get_last_lr()[0]

        val_metrics = None
        if val_loader:
            val_metrics = eval_epoch(model, val_loader, device)

        if val_metrics:
            score = val_metrics.get(best_key, float("nan"))
            if score != score:  # nan
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

        # Early stopping
        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(f"[EARLY STOP] No improvement for {args.early_stopping_patience} epochs. Stopping at epoch {epoch}.")
            if use_wandb:
                wandb.summary["early_stop_epoch"] = epoch
            break


    # final test with best
    if ckpt_best.exists():
        ckpt = torch.load(ckpt_best, map_location="cpu")
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