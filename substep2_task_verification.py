import os
import re
import csv
import json
import math
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# --------------------------
# Utils: seed / metrics
# --------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def parse_bool01(x) -> int:
    """
    Parse csv field to 0/1.
    Accepts: True/False, true/false, 1/0, yes/no, y/n, t/f.
    """
    if x is None:
        return 0
    if isinstance(x, (int, np.integer)):
        return int(x != 0)
    if isinstance(x, float):
        return int(x != 0.0)
    s = str(x).strip().strip('"').strip("'").lower()
    if s in {"true", "t", "yes", "y", "1"}:
        return 1
    if s in {"false", "f", "no", "n", "0", ""}:
        return 0
    # fallback: try numeric
    try:
        return int(float(s) != 0.0)
    except Exception:
        raise ValueError(f"Unrecognized boolean/numeric value for has_errors: {x!r}")

def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    """
    y_true: (N,) in {0,1}
    y_prob: (N,) in [0,1]
    """
    y_pred = (y_prob >= thr).astype(np.int64)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    out = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(acc),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }

    # Optional AUC / PR-AUC if sklearn exists
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        if len(np.unique(y_true)) == 2:
            out["auc"] = float(roc_auc_score(y_true, y_prob))
            out["pr_auc"] = float(average_precision_score(y_true, y_prob))
        else:
            out["auc"] = float("nan")
            out["pr_auc"] = float("nan")
    except Exception:
        out["auc"] = float("nan")
        out["pr_auc"] = float("nan")

    return out


# --------------------------
# Labels: derive video label from step_annotations.csv
# --------------------------
def load_video_labels_from_step_csv(step_csv: Path,
                                   id_col: str = "recording_id",
                                   err_col: str = "has_errors",
                                   correct_label: int = 1) -> Dict[str, int]:
    """
    Returns dict: video_id -> y (0/1)
    Default rule:
      video incorrect if ANY step has_errors==1
      y = 1(correct) if no errors else 0(incorrect)
    """
    by_video_err = {}
    with step_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if id_col not in reader.fieldnames or err_col not in reader.fieldnames:
            raise ValueError(f"CSV missing columns. Need '{id_col}' and '{err_col}'. Got: {reader.fieldnames}")

        for row in reader:
            vid = str(row[id_col])
            err = parse_bool01(row[err_col])  # tolerate "0"/"1"/"0.0"
            if vid not in by_video_err:
                by_video_err[vid] = 0
            by_video_err[vid] = max(by_video_err[vid], err)

    labels = {}
    for vid, has_err in by_video_err.items():
        is_correct = 1 if has_err == 0 else 0
        labels[vid] = is_correct if correct_label == 1 else (1 - is_correct)
    return labels


# --------------------------
# Dataset: read Substep1 outputs (per-video npz)
# --------------------------
def parse_recipe_id(video_id: str, mode: str = "prefix_underscore") -> str:
    """
    Default: '5_11' -> '5'
    Other option: regex:<pattern> with group(1)
    """
    if mode == "prefix_underscore":
        return video_id.split("_")[0]
    if mode.startswith("regex:"):
        pat = mode[len("regex:"):]
        m = re.match(pat, video_id)
        if not m:
            return video_id
        return m.group(1)
    return video_id


class VideoSeqDataset(Dataset):
    def __init__(self,
                 npz_paths: List[Path],
                 video_labels: Dict[str, int],
                 recipe_mode: str,
                 max_steps: int = 256,
                 min_steps: int = 1):
        """
        Each npz expected to contain:
          - embeddings: (N, D)
          - video_id (optional)
        """
        self.samples = []
        self.max_steps = max_steps
        self.recipe_mode = recipe_mode

        for p in npz_paths:
            vid = p.stem  # default: filename = {video_id}.npz
            if vid not in video_labels:
                continue
            # load embeddings shape quickly
            try:
                with np.load(p) as d:
                    if "embeddings" not in d:
                        continue
                    emb = d["embeddings"]
            except Exception:
                continue

            emb = np.asarray(emb)
            if emb.ndim != 2:
                continue
            if emb.shape[0] < min_steps:
                continue

            y = int(video_labels[vid])
            recipe_id = parse_recipe_id(vid, recipe_mode)
            self.samples.append((vid, recipe_id, p, y))

        if len(self.samples) == 0:
            raise RuntimeError("No usable samples found. Check labels mapping and npz directory.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vid, recipe_id, p, y = self.samples[idx]
        with np.load(p) as d:
            emb = np.asarray(d["embeddings"], dtype=np.float32)  # (N,D)

        # clip to max_steps
        if emb.shape[0] > self.max_steps:
            emb = emb[:self.max_steps]

        return {
            "video_id": vid,
            "recipe_id": recipe_id,
            "x": emb,          # (L,D)
            "y": np.int64(y)   # scalar
        }


def collate_pad(batch: List[Dict], pad_value: float = 0.0):
    """
    Pad sequences to max length in batch.
    Returns:
      x: (B, L, D) float32
      mask: (B, L) 1 for valid, 0 for pad
      y: (B,) int64
    """
    B = len(batch)
    lengths = [item["x"].shape[0] for item in batch]
    D = batch[0]["x"].shape[1]
    L = max(lengths)

    x = np.full((B, L, D), pad_value, dtype=np.float32)
    mask = np.zeros((B, L), dtype=np.float32)
    y = np.zeros((B,), dtype=np.int64)
    vids = []
    rids = []

    for i, item in enumerate(batch):
        li = item["x"].shape[0]
        x[i, :li] = item["x"]
        mask[i, :li] = 1.0
        y[i] = item["y"]
        vids.append(item["video_id"])
        rids.append(item["recipe_id"])

    return {
        "x": torch.from_numpy(x),         # (B,L,D)
        "mask": torch.from_numpy(mask),   # (B,L)
        "y": torch.from_numpy(y),         # (B,)
        "video_id": vids,
        "recipe_id": rids
    }


# --------------------------
# Model: Transformer + pooling + binary head
# --------------------------
class TransformerBinaryClassifier(nn.Module):
    def __init__(self,
                 in_dim: int,
                 d_model: int = 256,
                 nhead: int = 4,
                 dim_ff: int = 512,
                 dropout: float = 0.1,
                 num_layers: int = 1,
                 max_steps: int = 256):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model) if in_dim != d_model else nn.Identity()
        self.pos = nn.Embedding(max_steps, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)

        self.max_steps = max_steps

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: (B,L,in_dim)
        mask: (B,L) 1 valid, 0 pad
        """
        B, L, _ = x.shape
        if L > self.max_steps:
            x = x[:, :self.max_steps]
            mask = mask[:, :self.max_steps]
            L = self.max_steps

        h = self.proj(x)  # (B,L,d_model)

        pos_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = h + self.pos(pos_ids)

        # Transformer expects True for PAD positions
        pad_mask = (mask <= 0.0)  # (B,L) bool
        h = self.encoder(h, src_key_padding_mask=pad_mask)  # (B,L,d_model)

        # masked mean pooling
        mask_f = mask.unsqueeze(-1)  # (B,L,1)
        pooled = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

        pooled = self.dropout(pooled)
        logit = self.head(pooled).squeeze(-1)  # (B,)
        return logit


# --------------------------
# Train / eval per fold
# --------------------------
def run_one_fold(train_ds: VideoSeqDataset,
                 test_ds: VideoSeqDataset,
                 device: str,
                 epochs: int,
                 batch_size: int,
                 lr: float,
                 weight_decay: float,
                 d_model: int,
                 nhead: int,
                 dim_ff: int,
                 dropout: float,
                 seed: int) -> Dict[str, float]:

    # determine input dim from first sample
    sample0 = train_ds[0]
    in_dim = sample0["x"].shape[1]

    model = TransformerBinaryClassifier(
        in_dim=in_dim,
        d_model=d_model,
        nhead=nhead,
        dim_ff=dim_ff,
        dropout=dropout,
        num_layers=1,
        max_steps=train_ds.max_steps
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate_pad)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0, collate_fn=collate_pad)

    # train
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        for batch in train_loader:
            x = batch["x"].to(device)
            mask = batch["mask"].to(device)
            y = batch["y"].to(device).float()

            opt.zero_grad(set_to_none=True)
            logit = model(x, mask)
            loss = loss_fn(logit, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += float(loss.item()) * x.size(0)
            n += x.size(0)

        # optional: print training loss
        # print(f"  epoch {ep:02d} train_loss={total_loss/max(n,1):.4f}")

    # eval
    model.eval()
    all_true = []
    all_logit = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device)
            mask = batch["mask"].to(device)
            y = batch["y"].cpu().numpy().astype(np.int64)

            logit = model(x, mask).detach().cpu().numpy()
            all_true.append(y)
            all_logit.append(logit)

    y_true = np.concatenate(all_true, axis=0)
    y_logit = np.concatenate(all_logit, axis=0)
    y_prob = sigmoid(y_logit)

    return binary_metrics(y_true, y_prob, thr=0.5)


# --------------------------
# Main: Leave-one-out by recipe
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--substep1_dir", required=True, help="Directory of Substep1 outputs (per-video .npz with embeddings)")
    ap.add_argument("--step_annotations_csv", required=True, help="CSV containing recording_id + has_errors (and start/end columns ok)")
    ap.add_argument("--out_dir", required=True, help="Where to save results json")

    ap.add_argument("--id_col", default="recording_id")
    ap.add_argument("--err_col", default="has_errors")
    ap.add_argument("--recipe_mode", default="prefix_underscore",
                    help="prefix_underscore or regex:<pattern> with group(1)")

    ap.add_argument("--max_steps", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)

    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--dim_ff", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_seed(args.seed)

    substep1_dir = Path(args.substep1_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # labels: video_id -> 0/1 (1=correct)
    video_labels = load_video_labels_from_step_csv(
        Path(args.step_annotations_csv),
        id_col=args.id_col,
        err_col=args.err_col,
        correct_label=1
    )

    # collect npz paths
    npz_paths = sorted(substep1_dir.glob("*.npz"))
    ds_all = VideoSeqDataset(
        npz_paths=npz_paths,
        video_labels=video_labels,
        recipe_mode=args.recipe_mode,
        max_steps=args.max_steps
    )

    #debug:
    
    from collections import Counter, defaultdict
    import numpy as np

    # 1) Substep2 videos number
    ys = [y for (_, _, _, y) in ds_all.samples]
    print("USED videos:", len(ds_all.samples))
    print("USED label distribution:", Counter(ys))

    # 2)  how many videos per recipe
    rid_cnt = Counter([rid for (_, rid, _, _) in ds_all.samples])
    vals = np.array(list(rid_cnt.values()))
    print("recipes:", len(rid_cnt),
        "per-recipe count min/median/max:",
        vals.min(), np.median(vals), vals.max())

    # 3)  label distribution per recipe (check if each recipe is single-class)
    rid_label = defaultdict(list)
    for (_, rid, _, y) in ds_all.samples:
        rid_label[rid].append(y)

    single_class = 0
    for rid, lst in rid_label.items():
        c = Counter(lst)
        if len(c) == 1:
            single_class += 1
    print("recipes with SINGLE class in their videos:", single_class, "/", len(rid_label))

    #debug-end

    # list recipes
    recipes = sorted(list({r for (_, r, _, _) in ds_all.samples}))
    print(f"Total videos usable: {len(ds_all)}")
    print(f"Total recipes: {len(recipes)}")
    if len(recipes) <= 1:
        raise RuntimeError("Need at least 2 recipes for leave-one-out evaluation.")

    fold_results = {}
    metric_keys = ["precision", "recall", "f1", "accuracy", "auc", "pr_auc"]
    agg = {k: [] for k in metric_keys}

    # build index by recipe
    by_recipe_idx = {}
    for i, (vid, rid, p, y) in enumerate(ds_all.samples):
        by_recipe_idx.setdefault(rid, []).append(i)

    for rid in recipes:
        test_indices = set(by_recipe_idx.get(rid, []))
        train_indices = [i for i in range(len(ds_all.samples)) if i not in test_indices]
        test_indices = sorted(list(test_indices))

        if len(test_indices) == 0 or len(train_indices) == 0:
            print(f"[SKIP] recipe={rid} (train={len(train_indices)} test={len(test_indices)})")
            continue

        # create subset datasets
        train_paths = [ds_all.samples[i][2] for i in train_indices]
        test_paths = [ds_all.samples[i][2] for i in test_indices]

        train_ds = VideoSeqDataset(train_paths, video_labels, args.recipe_mode, max_steps=args.max_steps)
        test_ds = VideoSeqDataset(test_paths, video_labels, args.recipe_mode, max_steps=args.max_steps)

        res = run_one_fold(
            train_ds=train_ds,
            test_ds=test_ds,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            d_model=args.d_model,
            nhead=args.nhead,
            dim_ff=args.dim_ff,
            dropout=args.dropout,
            seed=args.seed
        )

        fold_results[rid] = res
        for k in metric_keys:
            if not math.isnan(res.get(k, float("nan"))):
                agg[k].append(res[k])

        print(f"[LOO] recipe={rid}  "
              f"acc={res['accuracy']:.3f} f1={res['f1']:.3f} "
              f"prec={res['precision']:.3f} rec={res['recall']:.3f} "
              f"auc={res['auc'] if not math.isnan(res['auc']) else 'nan'}")

    summary = {k: float(np.mean(v)) if len(v) > 0 else float("nan") for k, v in agg.items()}
    out = {
        "summary_mean": summary,
        "fold_results": fold_results,
        "settings": vars(args)
    }

    out_path = out_dir / "substep2_leave_one_out_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("\n=== Summary (mean over folds) ===")
    for k, v in summary.items():
        print(f"{k:>9}: {v:.4f}" if not math.isnan(v) else f"{k:>9}: nan")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
