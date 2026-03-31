import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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


# -------------------------
# Recordings parsing (CaptainCook4D recordings-combined.json)
# -------------------------
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
        db = data["database"]
    elif isinstance(data, dict):
        # fallback: assume whole dict is db
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

        # split（ Training/Validation/Test）
        subset = norm_subset(item.get("subset", ""))
        if subset is not None:
            split[subset].append(str(vid))

        # label： default use "any step has_error==True => error(0) else correct(1)"
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

    # keep deterministic order
    for k in split:
        split[k] = sorted(split[k])

    if not labels:
        raise RuntimeError(f"No labels parsed from {recordings_json}")

    return labels, split



# -------------------------
# Dataset: load per-video graph packages from Substep3
# -------------------------
@dataclass
class GraphSample:
    video_id: str
    edge_index: torch.Tensor      # (2,E) long
    x_text: torch.Tensor          # (V,Dt) float
    x_vis_node: torch.Tensor      # (V,Dv) float (aligned step feature per node; 0 if unmatched)
    y: int                        # 0/1


class GraphPTDataset(torch.utils.data.Dataset):
    """
    Reads Substep3 output .pt:
      required keys:
        - edge_index
        - x_text OR node_text_emb
        - step_x OR step_emb
        - node_to_step
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

        V = x_text.size(0)
        Dv = step_x.size(1)

        x_vis_node = torch.zeros((V, Dv), dtype=torch.float32)
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
        V = s.x_text.size(0)
        x_texts.append(s.x_text)
        x_viss.append(s.x_vis_node)
        ys.append(s.y)

        ei = s.edge_index
        if ei.numel() > 0:
            ei = ei + node_offset
        edge_indices.append(ei)

        batch_vecs.append(torch.full((V,), gi, dtype=torch.long))
        node_offset += V

    x_text = torch.cat(x_texts, dim=0)
    x_vis = torch.cat(x_viss, dim=0)
    edge_index = torch.cat(edge_indices, dim=1) if len(edge_indices) else torch.empty((2, 0), dtype=torch.long)
    batch_vec = torch.cat(batch_vecs, dim=0)
    y = torch.tensor(ys, dtype=torch.float32)

    return {"x_text": x_text, "x_vis": x_vis, "edge_index": edge_index, "batch": batch_vec, "y": y}


# -------------------------
# Model
# -------------------------
class NodeFusion(nn.Module):
    """
    h0 = LN( Wt(x_text) + Wv(x_vis_node) )
    x_vis_node is zero for unmatched nodes => OK.
    """
    def __init__(self, dim_text: int, dim_vis: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.text_proj = nn.Linear(dim_text, hidden_dim)
        self.vis_proj = nn.Linear(dim_vis, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x_text: torch.Tensor, x_vis: torch.Tensor) -> torch.Tensor:
        ht = self.drop(self.text_proj(x_text))
        hv = self.drop(self.vis_proj(x_vis))
        h = ht + hv
        return self.ln(h)


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

        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, msg)

        if self.aggr == "mean":
            deg = torch.zeros((x.size(0),), device=x.device, dtype=x.dtype)
            deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
            agg = agg / deg.clamp(min=1.0).unsqueeze(-1)

        out = self.self_lin(x) + agg
        out = F.relu(out)
        out = self.ln(out)
        return self.drop(out)


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
    ):
        super().__init__()
        self.fusion = NodeFusion(dim_text, dim_vis, hidden_dim, dropout=dropout)
        self.layers = nn.ModuleList([DiGraphConv(hidden_dim, dropout=dropout, aggr=aggr) for _ in range(num_layers)])

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_text: torch.Tensor, x_vis: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        h = self.fusion(x_text, x_vis)
        for layer in self.layers:
            h = h + layer(h, edge_index)  # residual
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        g = global_mean_pool(h, batch, num_graphs)
        logits = self.head(g).squeeze(-1)
        return logits


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

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc, "pr_auc": pr_auc}


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


# -------------------------
# Main
# -------------------------
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
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    # model
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--num_layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--aggr", choices=["sum", "mean"], default="mean")

    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--resume", default="", help="Path to checkpoint .pt")

    args = ap.parse_args()

    set_seed(args.seed)

    pt_dir = Path(args.graph_pt_dir)
    labels, split = load_recordings_combined(Path(args.recordings_json), label_rule=args.label_rule)

    # only keep vids that have .pt and label
    pt_vids = set(p.stem for p in pt_dir.glob("*.pt"))
    labeled_vids = set(labels.keys())
    usable = sorted(list(pt_vids & labeled_vids))
    if not usable:
        raise RuntimeError(f"No usable samples: check {pt_dir} and recordings_json labels")

    # build split from subset (preferred)
    train_vids = [v for v in split.get("train", []) if v in usable]
    val_vids = [v for v in split.get("val", []) if v in usable]
    test_vids = [v for v in split.get("test", []) if v in usable]

    # fallback: random split if subset split is empty
    if not train_vids and not val_vids and not test_vids:
        all_vids = usable[:]
        random.shuffle(all_vids)
        n = len(all_vids)
        n_train = int(n * args.train_ratio)
        n_val = int(n * args.val_ratio)
        train_vids = all_vids[:n_train]
        val_vids = all_vids[n_train:n_train + n_val]
        test_vids = all_vids[n_train + n_val:]

    if not train_vids:
        raise RuntimeError("Empty train set after filtering. Check pt files and subset in recordings_json.")

    train_ds = GraphPTDataset(pt_dir, train_vids, labels)
    val_ds = GraphPTDataset(pt_dir, val_vids, labels) if val_vids else None
    test_ds = GraphPTDataset(pt_dir, test_vids, labels) if test_vids else None

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_graph_samples,
        pin_memory=True,
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
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, device, optimizer, args.grad_clip)

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
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "best_score": best_score},
                ckpt_best
            )

        if val_metrics:
            print(f"Epoch {epoch:03d} | loss={tr_loss:.4f} | val={val_metrics} | best_score={best_score:.4f}")
        else:
            print(f"Epoch {epoch:03d} | loss={tr_loss:.4f} | best_score={best_score:.4f}")

    # final test with best
    if ckpt_best.exists():
        ckpt = torch.load(ckpt_best, map_location="cpu")
        model.load_state_dict(ckpt["model"])

    if test_loader:
        print("[FINAL TEST]", eval_epoch(model, test_loader, device))


if __name__ == "__main__":
    main()
