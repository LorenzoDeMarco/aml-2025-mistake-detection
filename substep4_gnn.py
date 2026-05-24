import argparse
import json
import os
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold


# -------------------------
# Utils
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def try_import_sklearn_auc():
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        return roc_auc_score, average_precision_score
    except Exception:
        return None, None


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(-x))


def find_first_key(d: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in d:
            return k
    return None


def format_metrics(metrics: Dict[str, float]) -> str:
    keys = ["accuracy", "precision", "recall", "f1", "auc", "pr_auc"]
    parts = [f"{k}={metrics[k]:.4f}" for k in keys if k in metrics]
    return ", ".join(parts)


# -------------------------
# Recordings parsing (CaptainCook4D recordings-combined.json)
# -------------------------
def load_recordings_combined(
    recordings_json: Path,
    label_rule: str = "any_step_error",
) -> Tuple[Dict[str, int], Dict[str, List[str]]]:
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

    def norm_subset(s: str) -> Optional[str]:
        s2 = str(s).strip().lower()
        if s2 in {"training", "train"}:
            return "train"
        if s2 in {"validation", "val", "valid"}:
            return "val"
        if s2 in {"test", "testing"}:
            return "test"
        return None

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


def intersect_split_with_available(
    split: Dict[str, List[str]],
    labels: Dict[str, int],
    pt_dir: Path,
) -> Tuple[Dict[str, List[str]], List[str]]:
    pt_vids = {p.stem for p in pt_dir.glob("*.pt")}
    labeled_vids = set(labels.keys())
    available = pt_vids & labeled_vids

    split_vids: Dict[str, List[str]] = {}
    for subset in ("train", "val", "test"):
        split_vids[subset] = sorted(vid for vid in split.get(subset, []) if vid in available)

    usable = sorted(available)
    return split_vids, usable


# -------------------------
# Dataset: load per-video graph packages from Substep3
# -------------------------
@dataclass
class GraphSample:
    video_id: str
    edge_index: torch.Tensor
    x_text: torch.Tensor
    x_vis_node: torch.Tensor
    y: int


class GraphPTDataset(torch.utils.data.Dataset):
    def __init__(self, pt_dir: Path, video_ids: List[str], labels: Dict[str, int]):
        self.pt_dir = pt_dir
        self.video_ids = video_ids
        self.labels = labels

        if not video_ids:
            raise ValueError("GraphPTDataset received an empty video id list")

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

        edge_index = pack["edge_index"].long()
        k_text = find_first_key(pack, ["x_text", "node_text_emb"])
        if k_text is None:
            raise KeyError(f"{vid}.pt missing x_text/node_text_emb")
        x_text = pack[k_text].float()

        k_step = find_first_key(pack, ["step_x", "step_emb"])
        if k_step is None:
            raise KeyError(f"{vid}.pt missing step_x/step_emb")
        step_x = pack[k_step].float()

        if "node_to_step" not in pack:
            raise KeyError(f"{vid}.pt missing node_to_step")
        node_to_step = pack["node_to_step"].long()

        v = x_text.size(0)
        dv = step_x.size(1)
        x_vis_node = torch.zeros((v, dv), dtype=torch.float32)
        matched = node_to_step >= 0
        if matched.any():
            node_idx = torch.nonzero(matched, as_tuple=False).squeeze(-1)
            step_idx = node_to_step[node_idx]
            x_vis_node[node_idx] = step_x[step_idx]

        y = int(self.labels[vid])
        return GraphSample(video_id=vid, edge_index=edge_index, x_text=x_text, x_vis_node=x_vis_node, y=y)


def collate_graph_samples(batch: List[GraphSample]) -> Dict[str, torch.Tensor]:
    x_texts, x_viss, edge_indices, batch_vecs, ys = [], [], [], [], []
    node_offset = 0
    for gi, s in enumerate(batch):
        v = s.x_text.size(0)
        x_texts.append(s.x_text)
        x_viss.append(s.x_vis_node)
        ys.append(s.y)

        ei = s.edge_index
        if ei.numel() > 0:
            ei = ei + node_offset
        edge_indices.append(ei)

        batch_vecs.append(torch.full((v,), gi, dtype=torch.long))
        node_offset += v

    x_text = torch.cat(x_texts, dim=0)
    x_vis = torch.cat(x_viss, dim=0)
    edge_index = torch.cat(edge_indices, dim=1) if edge_indices else torch.empty((2, 0), dtype=torch.long)
    batch_vec = torch.cat(batch_vecs, dim=0)
    y = torch.tensor(ys, dtype=torch.float32)

    return {"x_text": x_text, "x_vis": x_vis, "edge_index": edge_index, "batch": batch_vec, "y": y}


# -------------------------
# GNN layers
# -------------------------
class NodeFusion(nn.Module):
    def __init__(self, dim_text: int, dim_vis: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.text_proj = nn.Linear(dim_text, hidden_dim)
        self.vis_proj = nn.Linear(dim_vis, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x_text: torch.Tensor, x_vis: torch.Tensor) -> torch.Tensor:
        ht = self.drop(self.text_proj(x_text))
        hv = self.drop(self.vis_proj(x_vis))
        return self.ln(ht + hv)


class DiGraphConv(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1, aggr: str = "mean"):
        super().__init__()
        assert aggr in {"sum", "mean"}
        self.aggr = aggr
        self.msg = nn.Linear(dim, dim, bias=False)
        self.self_lin = nn.Linear(dim, dim, bias=True)
        self.ln = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        del batch
        if edge_index.numel() == 0:
            out = F.relu(self.self_lin(x))
            return self.ln(self.drop(out))

        src, dst = edge_index[0], edge_index[1]
        msg = self.msg(x[src])

        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, msg)

        if self.aggr == "mean":
            deg = torch.zeros((x.size(0),), device=x.device, dtype=x.dtype)
            deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
            agg = agg / deg.clamp(min=1.0).unsqueeze(-1)

        out = F.relu(self.self_lin(x) + agg)
        return self.ln(self.drop(out))


def topological_order_from_edges(num_nodes: int, edge_index: torch.Tensor) -> List[int]:
    """Kahn topological sort for one graph. Falls back to identity order on cycles."""
    if num_nodes == 0:
        return []
    if edge_index.numel() == 0:
        return list(range(num_nodes))

    indeg = [0] * num_nodes
    adj: List[List[int]] = [[] for _ in range(num_nodes)]
    edges = edge_index.t().tolist()
    for src, dst in edges:
        if 0 <= src < num_nodes and 0 <= dst < num_nodes and src != dst:
            adj[src].append(dst)
            indeg[dst] += 1

    queue = deque(i for i in range(num_nodes) if indeg[i] == 0)
    order: List[int] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for nxt in adj[node]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)

    if len(order) != num_nodes:
        return list(range(num_nodes))
    return order


class DAGNNConv(nn.Module):
    """
    Lightweight DAGNN-style layer: propagate only from predecessors in topological order.
    Designed for CaptainCook task graphs (directed acyclic graphs).
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.self_lin = nn.Linear(dim, dim, bias=True)
        self.pred_lin = nn.Linear(dim, dim, bias=False)
        self.ln = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def _forward_single_graph(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        order = topological_order_from_edges(num_nodes, edge_index.cpu())

        preds: List[List[int]] = [[] for _ in range(num_nodes)]
        if edge_index.numel() > 0:
            for src, dst in edge_index.t().tolist():
                if 0 <= src < num_nodes and 0 <= dst < num_nodes:
                    preds[dst].append(src)

        h = torch.zeros_like(x)
        for node in order:
            if preds[node]:
                parent_h = torch.stack([h[p] for p in preds[node]], dim=0).mean(dim=0)
                h[node] = F.relu(self.self_lin(x[node]) + self.pred_lin(parent_h))
            else:
                h[node] = F.relu(self.self_lin(x[node]))
        return self.ln(self.drop(h))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if batch.numel() == 0:
            return self._forward_single_graph(x, edge_index)

        num_graphs = int(batch.max().item()) + 1
        outputs = []
        for graph_id in range(num_graphs):
            node_mask = batch == graph_id
            node_ids = torch.nonzero(node_mask, as_tuple=False).squeeze(-1)
            x_g = x[node_ids]
            if edge_index.numel() == 0:
                ei_g = edge_index
            else:
                edge_mask = (batch[edge_index[0]] == graph_id) & (batch[edge_index[1]] == graph_id)
                ei_g = edge_index[:, edge_mask]
                local_map = torch.full((x.size(0),), -1, dtype=torch.long, device=x.device)
                local_map[node_ids] = torch.arange(node_ids.numel(), device=x.device)
                ei_g = local_map[ei_g]
            outputs.append(self._forward_single_graph(x_g, ei_g))
        return torch.cat(outputs, dim=0)


def global_mean_pool(x: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    dim = x.size(-1)
    out = torch.zeros((num_graphs, dim), device=x.device, dtype=x.dtype)
    out.index_add_(0, batch, x)
    cnt = torch.zeros((num_graphs,), device=x.device, dtype=x.dtype)
    cnt.index_add_(0, batch, torch.ones_like(batch, dtype=x.dtype))
    return out / cnt.clamp(min=1.0).unsqueeze(-1)


class GraphClassifier(nn.Module):
    def __init__(
        self,
        dim_text: int,
        dim_vis: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.2,
        aggr: str = "mean",
        gnn_layer: str = "dagnn",
    ):
        super().__init__()
        assert gnn_layer in {"dagnn", "digraph"}
        self.gnn_layer = gnn_layer
        self.fusion = NodeFusion(dim_text, dim_vis, hidden_dim, dropout=dropout)

        if gnn_layer == "dagnn":
            self.layers = nn.ModuleList([DAGNNConv(hidden_dim, dropout=dropout) for _ in range(num_layers)])
        else:
            self.layers = nn.ModuleList([
                DiGraphConv(hidden_dim, dropout=dropout, aggr=aggr) for _ in range(num_layers)
            ])

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x_text: torch.Tensor,
        x_vis: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        h = self.fusion(x_text, x_vis)
        for layer in self.layers:
            if self.gnn_layer == "dagnn":
                h = h + layer(h, edge_index, batch)
            else:
                h = h + layer(h, edge_index, batch)
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        g = global_mean_pool(h, batch, num_graphs)
        return self.head(g).squeeze(-1)


# -------------------------
# Train / Eval functions
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
        y = batch["y"].to(device)

        logits = model(x_text, x_vis, edge_index, bvec)
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
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "pr_auc": pr_auc,
        "num_samples": int(y.numel()),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def train_one_epoch(model: nn.Module, loader, device: torch.device, optimizer, grad_clip: float) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0
    for batch in loader:
        x_text = batch["x_text"].to(device)
        x_vis = batch["x_vis"].to(device)
        edge_index = batch["edge_index"].to(device)
        bvec = batch["batch"].to(device)
        y = batch["y"].to(device)

        logits = model(x_text, x_vis, edge_index, bvec)
        loss = F.binary_cross_entropy_with_logits(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        total_n += bs
    return total_loss / max(1, total_n)


def make_loader(dataset, batch_size, shuffle, num_workers):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_graph_samples,
        pin_memory=True,
    )


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    optimizer,
    epochs: int,
    grad_clip: float,
) -> Dict[str, torch.Tensor]:
    best_score = -1e9
    best_state = None
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, device, optimizer, grad_clip)
        if val_loader is not None:
            val_metrics = eval_epoch(model, val_loader, device)
            score = val_metrics.get("auc", val_metrics.get("f1", -1e9))
            if score > best_score:
                best_score = score
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if epoch % 10 == 0 or epoch == epochs:
                print(
                    f"Epoch {epoch:03d}/{epochs} | loss={train_loss:.4f} | "
                    f"val: {format_metrics(val_metrics)}"
                )
        else:
            if -train_loss > best_score:
                best_score = -train_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if epoch % 10 == 0 or epoch == epochs:
                print(f"Epoch {epoch:03d}/{epochs} | loss={train_loss:.4f}")

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    return best_state


def run_official_split(args, pt_dir: Path, labels: Dict[str, int], split: Dict[str, List[str]]):
    split_vids, usable = intersect_split_with_available(split, labels, pt_dir)
    if not usable:
        raise RuntimeError("No usable samples after intersecting pt files, labels, and recordings.json")

    train_vids = split_vids["train"]
    val_vids = split_vids["val"]
    test_vids = split_vids["test"]

    print("Split sizes (official CaptainCook subset):")
    print(f"  train={len(train_vids)}, val={len(val_vids)}, test={len(test_vids)}, total_available={len(usable)}")

    if not train_vids:
        raise RuntimeError("Official train split is empty after filtering to available pt files")
    if not test_vids:
        raise RuntimeError("Official test split is empty after filtering to available pt files")

    sample0 = torch.load(pt_dir / f"{usable[0]}.pt", map_location="cpu")
    k_text = find_first_key(sample0, ["x_text", "node_text_emb"])
    k_step = find_first_key(sample0, ["step_x", "step_emb"])
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
        gnn_layer=args.gnn_layer,
    ).to(device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_only:
        if not args.resume:
            raise ValueError("--eval_only requires --resume pointing to a trained checkpoint")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {args.resume}")
    else:
        train_ds = GraphPTDataset(pt_dir, train_vids, labels)
        val_ds = GraphPTDataset(pt_dir, val_vids, labels) if val_vids else None
        train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers)
        val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers) if val_ds else None

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_state = train_model(
            model, train_loader, val_loader, device, optimizer, args.epochs, args.grad_clip
        )
        model.load_state_dict(best_state)
        torch.save(best_state, out_dir / "best.pt")
        print(f"Saved checkpoint to {out_dir / 'best.pt'}")

    test_ds = GraphPTDataset(pt_dir, test_vids, labels)
    test_loader = make_loader(test_ds, args.batch_size, False, args.num_workers)
    test_metrics = eval_epoch(model, test_loader, device)

    val_metrics = None
    if val_vids:
        val_ds = GraphPTDataset(pt_dir, val_vids, labels)
        val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers)
        val_metrics = eval_epoch(model, val_loader, device)

    print("\n===== Final Test Results (official CaptainCook test split) =====")
    print(format_metrics(test_metrics))
    print(
        f"samples={test_metrics['num_samples']} | "
        f"tp={test_metrics['tp']} tn={test_metrics['tn']} fp={test_metrics['fp']} fn={test_metrics['fn']}"
    )
    if val_metrics is not None:
        print("\nValidation reference:")
        print(format_metrics(val_metrics))

    results = {
        "split_mode": "official",
        "gnn_layer": args.gnn_layer,
        "label_rule": args.label_rule,
        "train_size": len(train_vids),
        "val_size": len(val_vids),
        "test_size": len(test_vids),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_dir / 'results.json'}")


def run_kfold(args, pt_dir: Path, labels: Dict[str, int]):
    pt_vids = set(p.stem for p in pt_dir.glob("*.pt"))
    labeled_vids = set(labels.keys())
    usable = sorted(list(pt_vids & labeled_vids))
    if not usable:
        raise RuntimeError("No usable samples: check pt_dir and recordings_json labels")

    sample0 = torch.load(pt_dir / f"{usable[0]}.pt", map_location="cpu")
    k_text = find_first_key(sample0, ["x_text", "node_text_emb"])
    k_step = find_first_key(sample0, ["step_x", "step_emb"])
    dim_text = int(sample0[k_text].shape[1])
    dim_vis = int(sample0[k_step].shape[1])
    device = torch.device(args.device)

    kfold = KFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    all_test_metrics = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(usable)):
        print(f"\n=== Fold {fold + 1}/{args.kfold} ===")
        train_vids = [usable[i] for i in train_idx]
        test_vids = [usable[i] for i in test_idx]
        n_val = max(1, int(len(train_vids) * 0.1))
        val_vids = train_vids[:n_val]
        train_vids = train_vids[n_val:]

        model = GraphClassifier(
            dim_text=dim_text,
            dim_vis=dim_vis,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            aggr=args.aggr,
            gnn_layer=args.gnn_layer,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        train_loader = make_loader(GraphPTDataset(pt_dir, train_vids, labels), args.batch_size, True, args.num_workers)
        val_loader = make_loader(GraphPTDataset(pt_dir, val_vids, labels), args.batch_size, False, args.num_workers)
        test_loader = make_loader(GraphPTDataset(pt_dir, test_vids, labels), args.batch_size, False, args.num_workers)

        best_state = train_model(model, train_loader, val_loader, device, optimizer, args.epochs, args.grad_clip)
        model.load_state_dict(best_state)
        all_test_metrics.append(eval_epoch(model, test_loader, device))
        print(f"Fold {fold + 1} test: {format_metrics(all_test_metrics[-1])}")

    avg_metrics = {k: float(np.mean([m[k] for m in all_test_metrics])) for k in all_test_metrics[0] if k not in {"tp", "tn", "fp", "fn", "num_samples"}}
    std_metrics = {k: float(np.std([m[k] for m in all_test_metrics])) for k in avg_metrics}

    print("\n===== K-Fold Summary (mean ± std across folds) =====")
    for k in avg_metrics:
        print(f"{k}: {avg_metrics[k]:.4f} ± {std_metrics[k]:.4f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "cv_results.json").open("w", encoding="utf-8") as f:
        json.dump({"mean": avg_metrics, "std": std_metrics, "all_folds": all_test_metrics}, f, indent=2)
    print(f"Results saved to {out_dir / 'cv_results.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_pt_dir", required=True)
    ap.add_argument("--recordings_json", required=True)
    ap.add_argument("--label_rule", choices=["any_step_error", "video_has_error"], default="any_step_error")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--split_mode",
        choices=["official", "kfold"],
        default="official",
        help="official: CaptainCook train/val/test from recordings.json; kfold: legacy random K-fold",
    )
    ap.add_argument(
        "--gnn_layer",
        choices=["dagnn", "digraph"],
        default="dagnn",
        help="GNN layer type: DAGNN (default, for DAG task graphs) or legacy DiGraphConv",
    )
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--num_layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--aggr", choices=["sum", "mean"], default="sum")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--resume", default="")
    ap.add_argument("--kfold", type=int, default=5, help="Only used when --split_mode kfold")
    args = ap.parse_args()

    set_seed(args.seed)

    pt_dir = Path(args.graph_pt_dir)
    labels, split = load_recordings_combined(Path(args.recordings_json), label_rule=args.label_rule)

    if args.split_mode == "official":
        run_official_split(args, pt_dir, labels, split)
    else:
        run_kfold(args, pt_dir, labels)


if __name__ == "__main__":
    main()
