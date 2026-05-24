import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

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
from tqdm import tqdm


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def recipe_id_from_video_id(video_id: str) -> str:
    return video_id.split("_")[0]


def load_video_labels(recordings_json: Path, label_rule: str = "any_step_error") -> Dict[str, int]:
    """
    Return video_id -> label.
    1 = correct execution, 0 = incorrect (contains at least one error step by default).
    """
    with recordings_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    db = data["database"] if "database" in data else data

    labels: Dict[str, int] = {}
    for vid, item in db.items():
        if not isinstance(item, dict):
            continue
        anns = item.get("annotations", [])
        any_step_error = False
        if isinstance(anns, list):
            for ann in anns:
                if isinstance(ann, dict) and bool(ann.get("has_error", False)):
                    any_step_error = True
                    break

        if label_rule == "any_step_error":
            labels[str(vid)] = 0 if any_step_error else 1
        elif label_rule == "video_has_error":
            labels[str(vid)] = 0 if bool(item.get("has_error", False)) else 1
        else:
            raise ValueError(f"Unknown label_rule={label_rule}")
    return labels


def load_step_embeddings(substep1_dir: Path) -> Dict[str, np.ndarray]:
    """Load (S, D) step embeddings from substep1_4 npz files."""
    embeddings: Dict[str, np.ndarray] = {}
    for npz_path in sorted(substep1_dir.glob("*.npz")):
        with np.load(npz_path, allow_pickle=False) as data:
            if "embeddings" in data.files:
                arr = np.asarray(data["embeddings"], dtype=np.float32)
            elif "step_features" in data.files:
                arr = np.asarray(data["step_features"], dtype=np.float32)
            else:
                raise KeyError(f"{npz_path.name} missing embeddings/step_features, keys={data.files}")
        if arr.ndim != 2:
            raise ValueError(f"{npz_path.name}: expected 2D embeddings, got {arr.shape}")
        embeddings[npz_path.stem] = arr
    return embeddings


def load_aligned_dataset(
    substep1_dir: Path,
    recordings_json: Path,
    label_rule: str,
) -> Tuple[List[str], List[np.ndarray], List[int], Dict[str, List[str]]]:
    labels = load_video_labels(recordings_json, label_rule=label_rule)
    embeddings = load_step_embeddings(substep1_dir)

    video_ids = sorted(set(embeddings.keys()) & set(labels.keys()))
    if not video_ids:
        raise RuntimeError(
            f"No overlapping videos between {substep1_dir} and {recordings_json}"
        )

    x_list = [embeddings[vid] for vid in video_ids]
    y_list = [labels[vid] for vid in video_ids]

    recipe_to_videos: Dict[str, List[str]] = defaultdict(list)
    for vid in video_ids:
        recipe_to_videos[recipe_id_from_video_id(vid)].append(vid)
    for recipe_id in recipe_to_videos:
        recipe_to_videos[recipe_id] = sorted(recipe_to_videos[recipe_id])

    return video_ids, x_list, y_list, dict(recipe_to_videos)


class StepSequenceDataset(Dataset):
    def __init__(self, video_ids: List[str], x_list: List[np.ndarray], y_list: List[int], max_steps: int):
        self.video_ids = video_ids
        self.x_list = x_list
        self.y_list = y_list
        self.max_steps = max_steps

    def __len__(self) -> int:
        return len(self.x_list)

    def __getitem__(self, idx: int):
        x = self.x_list[idx]
        y = self.y_list[idx]
        s = x.shape[0]

        if s < self.max_steps:
            pad = np.zeros((self.max_steps - s, x.shape[1]), dtype=np.float32)
            x = np.vstack([x, pad])
        else:
            x = x[: self.max_steps]

        mask = np.zeros(self.max_steps, dtype=np.bool_)
        mask[: min(s, self.max_steps)] = True

        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(mask),
            torch.tensor(y, dtype=torch.long),
        )


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        num_layers: int,
        hidden_dim: int,
        dropout: float,
        max_steps: int,
    ):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_steps, emb_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embed[:, : x.size(1), :]
        x = self.transformer(x, src_key_padding_mask=~mask)
        masked_x = x * mask.unsqueeze(-1).float()
        pooled = masked_x.sum(dim=1) / mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
        return self.classifier(pooled)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, mask, y in loader:
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x, mask), y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for x, mask, y in loader:
        x, mask = x.to(device), mask.to(device)
        logits = model(x, mask)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(y.numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
    return all_preds, all_labels, all_probs


def compute_metrics(y_true: List[int], y_pred: List[int], y_prob: List[float]) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float("nan")
    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except ValueError:
        pr_auc = float("nan")
    return {
        "accuracy": float(acc),
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


def format_metrics(metrics: Dict[str, float]) -> str:
    return (
        f"accuracy={metrics['accuracy']:.4f}, precision={metrics['precision']:.4f}, "
        f"recall={metrics['recall']:.4f}, f1={metrics['f1']:.4f}, "
        f"auc={metrics['auc']:.4f}, pr_auc={metrics['pr_auc']:.4f}"
    )


def leave_one_recipe_out(
    video_ids: List[str],
    x_list: List[np.ndarray],
    y_list: List[int],
    recipe_to_videos: Dict[str, List[str]],
    emb_dim: int,
    max_steps: int,
    device: torch.device,
    args,
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    id_to_idx = {vid: i for i, vid in enumerate(video_ids)}
    recipe_ids = sorted(recipe_to_videos.keys())

    all_truth: List[int] = []
    all_pred: List[int] = []
    all_probs: List[float] = []
    fold_results: List[Dict[str, object]] = []

    pbar = tqdm(recipe_ids, desc="Leave-One-Recipe-Out", unit="recipe")
    for test_recipe in pbar:
        test_vids = recipe_to_videos[test_recipe]
        train_vids = [vid for vid in video_ids if vid not in set(test_vids)]

        train_X = [x_list[id_to_idx[vid]] for vid in train_vids]
        train_y = [y_list[id_to_idx[vid]] for vid in train_vids]
        test_X = [x_list[id_to_idx[vid]] for vid in test_vids]
        test_y = [y_list[id_to_idx[vid]] for vid in test_vids]

        train_set = StepSequenceDataset(train_vids, train_X, train_y, max_steps)
        test_set = StepSequenceDataset(test_vids, test_X, test_y, max_steps)

        train_loader = DataLoader(
            train_set,
            batch_size=min(args.batch_size, len(train_set)),
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

        model = TransformerClassifier(
            emb_dim=emb_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            max_steps=max_steps,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        last_loss = float("nan")
        for _ in range(args.epochs):
            last_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        preds, labels, probs = evaluate(model, test_loader, device)
        all_pred.extend(preds)
        all_truth.extend(labels)
        all_probs.extend(probs)

        fold_metrics = compute_metrics(labels, preds, probs)
        fold_results.append(
            {
                "test_recipe_id": test_recipe,
                "num_train_videos": len(train_vids),
                "num_test_videos": len(test_vids),
                "test_video_ids": test_vids,
                "last_train_loss": last_loss,
                "metrics": fold_metrics,
            }
        )
        pbar.set_postfix(
            {
                "recipe": test_recipe,
                "test_videos": len(test_vids),
                "loss": f"{last_loss:.4f}" if not np.isnan(last_loss) else "N/A",
            }
        )

    overall = compute_metrics(all_truth, all_pred, all_probs)
    return overall, fold_results


def main():
    ap = argparse.ArgumentParser(
        description="Substep2: Transformer baseline for recipe execution verification (leave-one-recipe-out)."
    )
    ap.add_argument(
        "--substep1_dir",
        default="./data/substep1_4_step_localization",
        help="Directory with substep1_4 per-video npz (embeddings)",
    )
    ap.add_argument(
        "--recordings_json",
        default="./data/substep1_1_actionformer_annotations/combined/recordings.json",
        help="CaptainCook recordings json with has_error labels",
    )
    ap.add_argument("--output_dir", default="./data/substep2_transformer_baseline")
    ap.add_argument("--label_rule", choices=["any_step_error", "video_has_error"], default="any_step_error")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max_steps", type=int, default=0, help="Pad/truncate length; 0 = auto from data")
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    substep1_dir = Path(args.substep1_dir)
    recordings_json = Path(args.recordings_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_ids, x_list, y_list, recipe_to_videos = load_aligned_dataset(
        substep1_dir, recordings_json, args.label_rule
    )
    emb_dim = int(x_list[0].shape[1])
    data_max_steps = max(x.shape[0] for x in x_list)
    max_steps = args.max_steps if args.max_steps > 0 else data_max_steps

    print(f"Device: {device}")
    print(f"Videos: {len(video_ids)}, recipes: {len(recipe_to_videos)}, emb_dim: {emb_dim}, max_steps: {max_steps}")

    overall, fold_results = leave_one_recipe_out(
        video_ids, x_list, y_list, recipe_to_videos,
        emb_dim, max_steps, device, args,
    )

    print("\n===== Final Leave-One-Recipe-Out Results =====")
    print(format_metrics(overall))
    print(
        f"samples={overall['num_samples']} | "
        f"tp={overall['tp']} tn={overall['tn']} fp={overall['fp']} fn={overall['fn']}"
    )
    print("Label: 1=correct execution, 0=incorrect execution")

    results = {
        "eval_mode": "leave_one_recipe_out",
        "substep1_dir": str(substep1_dir),
        "recordings_json": str(recordings_json),
        "label_rule": args.label_rule,
        "num_videos": len(video_ids),
        "num_recipes": len(recipe_to_videos),
        "emb_dim": emb_dim,
        "max_steps": max_steps,
        "overall_metrics": overall,
        "folds": fold_results,
    }
    out_path = output_dir / "substep2_leave_one_out_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
