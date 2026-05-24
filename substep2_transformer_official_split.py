"""
Substep 2 (official split): Transformer task-verification baseline.

Same model and features as substep2_transformer_baseline.py, but uses CaptainCook
official train / val / test from recordings.json — comparable to substep4_gnn.py.
"""
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_recordings_combined(
    recordings_json: Path,
    label_rule: str = "any_step_error",
) -> Tuple[Dict[str, int], Dict[str, List[str]]]:
    with recordings_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    db = data["database"] if isinstance(data, dict) and "database" in data else data

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
        any_step_error = any(
            isinstance(a, dict) and bool(a.get("has_error", False)) for a in anns
        )
        if label_rule == "any_step_error":
            labels[str(vid)] = 0 if any_step_error else 1
        elif label_rule == "video_has_error":
            labels[str(vid)] = 0 if bool(item.get("has_error", False)) else 1
        else:
            raise ValueError(f"Unknown label_rule={label_rule}")

    for key in split:
        split[key] = sorted(split[key])
    return labels, split


def load_step_embeddings(substep1_dir: Path) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for npz_path in sorted(substep1_dir.glob("*.npz")):
        with np.load(npz_path, allow_pickle=False) as data:
            if "embeddings" in data.files:
                arr = np.asarray(data["embeddings"], dtype=np.float32)
            elif "step_features" in data.files:
                arr = np.asarray(data["step_features"], dtype=np.float32)
            else:
                raise KeyError(f"{npz_path.name} missing embeddings/step_features")
        if arr.ndim != 2:
            raise ValueError(f"{npz_path.name}: expected 2D embeddings, got {arr.shape}")
        out[npz_path.stem] = arr
    return out


def intersect_official_split(
    split: Dict[str, List[str]],
    labels: Dict[str, int],
    substep1_dir: Path,
) -> Tuple[Dict[str, List[str]], List[str]]:
    available = {p.stem for p in substep1_dir.glob("*.npz")} & set(labels.keys())
    split_vids = {
        subset: sorted(vid for vid in split.get(subset, []) if vid in available)
        for subset in ("train", "val", "test")
    }
    return split_vids, sorted(available)


def subset_data(
    video_ids: List[str],
    x_list: List[np.ndarray],
    y_list: List[int],
    selected: List[str],
) -> Tuple[List[str], List[np.ndarray], List[int]]:
    idx = {vid: i for i, vid in enumerate(video_ids)}
    vids = sorted(selected)
    return vids, [x_list[idx[v]] for v in vids], [y_list[idx[v]] for v in vids]


class StepSequenceDataset(Dataset):
    def __init__(self, video_ids: List[str], x_list: List[np.ndarray], y_list: List[int], max_steps: int):
        if not video_ids:
            raise ValueError("Empty dataset")
        self.video_ids = video_ids
        self.x_list = x_list
        self.y_list = y_list
        self.max_steps = max_steps

    def __len__(self) -> int:
        return len(self.x_list)

    def __getitem__(self, idx: int):
        x = self.x_list[idx]
        s = x.shape[0]
        if s < self.max_steps:
            x = np.vstack([x, np.zeros((self.max_steps - s, x.shape[1]), dtype=np.float32)])
        else:
            x = x[: self.max_steps]
        mask = np.zeros(self.max_steps, dtype=np.bool_)
        mask[: min(s, self.max_steps)] = True
        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(mask),
            torch.tensor(self.y_list[idx], dtype=torch.long),
        )


class TransformerClassifier(nn.Module):
    def __init__(self, emb_dim, num_heads, num_layers, hidden_dim, dropout, max_steps):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_steps, emb_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embed[:, : x.size(1), :]
        x = self.transformer(x, src_key_padding_mask=~mask)
        masked = x * mask.unsqueeze(-1).float()
        pooled = masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
        return self.classifier(pooled)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for x, mask, y in loader:
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x, mask), y)
        loss.backward()
        optimizer.step()
        total += float(loss.item())
    return total / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels, probs = [], [], []
    for x, mask, y in loader:
        x, mask = x.to(device), mask.to(device)
        logits = model(x, mask)
        prob = torch.softmax(logits, dim=1)[:, 1]
        preds.extend(logits.argmax(dim=1).cpu().tolist())
        labels.extend(y.numpy().tolist())
        probs.extend(prob.cpu().numpy().tolist())
    return labels, preds, probs


def compute_metrics(y_true, y_pred, y_prob):
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float("nan")
    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except ValueError:
        pr_auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": auc,
        "pr_auc": pr_auc,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "num_samples": len(y_true),
    }


def format_metrics(m: Dict[str, float]) -> str:
    return (
        f"accuracy={m['accuracy']:.4f}, precision={m['precision']:.4f}, "
        f"recall={m['recall']:.4f}, f1={m['f1']:.4f}, "
        f"auc={m['auc']:.4f}, pr_auc={m['pr_auc']:.4f}"
    )


def make_loader(dataset, batch_size, shuffle, num_workers, device):
    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def main():
    ap = argparse.ArgumentParser(
        description="Substep2 Transformer baseline on official CaptainCook train/val/test."
    )
    ap.add_argument("--substep1_dir", default="./data/substep1_4_step_localization")
    ap.add_argument(
        "--recordings_json",
        default="./data/substep1_1_actionformer_annotations/combined/recordings.json",
    )
    ap.add_argument("--output_dir", default="./data/substep2_transformer_official")
    ap.add_argument("--label_rule", choices=["any_step_error", "video_has_error"], default="any_step_error")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--resume", default="")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    substep1_dir = Path(args.substep1_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels, split = load_recordings_combined(Path(args.recordings_json), args.label_rule)
    split_vids, usable = intersect_official_split(split, labels, substep1_dir)
    if not usable:
        raise RuntimeError("No videos with both npz embeddings and labels")

    embeddings = load_step_embeddings(substep1_dir)
    all_ids = sorted(set(usable))
    x_all = [embeddings[v] for v in all_ids]
    y_all = [labels[v] for v in all_ids]

    train_ids, train_x, train_y = subset_data(all_ids, x_all, y_all, split_vids["train"])
    val_ids, val_x, val_y = subset_data(all_ids, x_all, y_all, split_vids["val"])
    test_ids, test_x, test_y = subset_data(all_ids, x_all, y_all, split_vids["test"])

    if not train_ids:
        raise RuntimeError("Official train split is empty after filtering")
    if not test_ids:
        raise RuntimeError("Official test split is empty after filtering")

    emb_dim = int(x_all[0].shape[1])
    max_steps = args.max_steps if args.max_steps > 0 else max(x.shape[0] for x in x_all)

    print("Official split (videos with substep1_4 npz):")
    print(f"  train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    print(f"  emb_dim={emb_dim}, max_steps={max_steps}, device={device}")

    model = TransformerClassifier(
        emb_dim, args.num_heads, args.num_layers, args.hidden_dim, args.dropout, max_steps
    ).to(device)

    if args.eval_only:
        if not args.resume:
            raise ValueError("--eval_only requires --resume")
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Loaded checkpoint: {args.resume}")
    else:
        train_loader = make_loader(
            StepSequenceDataset(train_ids, train_x, train_y, max_steps),
            args.batch_size, True, args.num_workers, device,
        )
        val_loader = (
            make_loader(
                StepSequenceDataset(val_ids, val_x, val_y, max_steps),
                args.batch_size, False, args.num_workers, device,
            )
            if val_ids
            else None
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        best_score = -1e9
        best_state = None
        for epoch in range(1, args.epochs + 1):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            if val_loader is not None:
                val_m = compute_metrics(*evaluate(model, val_loader, device))  # y_true, y_pred, y_prob
                score = val_m.get("auc", val_m.get("f1", -1e9))
                if score > best_score:
                    best_score = score
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                if epoch % 10 == 0 or epoch == args.epochs:
                    print(f"Epoch {epoch:03d} | loss={loss:.4f} | val: {format_metrics(val_m)}")
            else:
                if -loss > best_score:
                    best_score = -loss
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                if epoch % 10 == 0 or epoch == args.epochs:
                    print(f"Epoch {epoch:03d} | loss={loss:.4f}")

        model.load_state_dict(best_state)
        torch.save(best_state, output_dir / "best.pt")
        print(f"Saved checkpoint: {output_dir / 'best.pt'}")

    test_loader = make_loader(
        StepSequenceDataset(test_ids, test_x, test_y, max_steps),
        args.batch_size, False, args.num_workers, device,
    )
    test_metrics = compute_metrics(*evaluate(model, test_loader, device))

    val_metrics = None
    if val_ids:
        val_loader = make_loader(
            StepSequenceDataset(val_ids, val_x, val_y, max_steps),
            args.batch_size, False, args.num_workers, device,
        )
        val_metrics = compute_metrics(*evaluate(model, val_loader, device))

    print("\n===== Final Test Results (official CaptainCook test split) =====")
    print(format_metrics(test_metrics))
    print(
        f"samples={test_metrics['num_samples']} | "
        f"tp={test_metrics['tp']} tn={test_metrics['tn']} "
        f"fp={test_metrics['fp']} fn={test_metrics['fn']}"
    )
    print("Label: 1=correct execution, 0=incorrect execution")
    if val_metrics is not None:
        print("\nValidation reference:")
        print(format_metrics(val_metrics))

    results = {
        "eval_mode": "official_captaincook_split",
        "model": "transformer",
        "substep1_dir": str(substep1_dir),
        "recordings_json": str(args.recordings_json),
        "label_rule": args.label_rule,
        "train_size": len(train_ids),
        "val_size": len(val_ids),
        "test_size": len(test_ids),
        "emb_dim": emb_dim,
        "max_steps": max_steps,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    out_json = output_dir / "results.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_json}")


if __name__ == "__main__":
    main()
