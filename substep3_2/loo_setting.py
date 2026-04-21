from pathlib import Path
import numpy as np
from typing import Dict
import argparse
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import csv

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def step_label(label_csv, target_id):
    label_list = []
    with open(label_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            recording_id = row[0]
            if recording_id == target_id:
                step_id = row[1]
                is_error = 1 if row[5] == "True" else 0
                label_list.append((recording_id, step_id, is_error))
    return label_list


def CSV_read(label_csv):
    recipe_dict = {}
    all_ids = set()

    with open(label_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            all_ids.add(row[0])  # unique recording_id, e.g. "1_7", "2_3"

    for rid in all_ids:
        labels = step_label(label_csv, rid)
        has_error = any(val == 1 for (_, _, val) in labels)
        recipe_dict[rid] = 1 if has_error else 0

    return recipe_dict  # Dict: video_id -> binary label (0=correct, 1=incorrect)


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= thr).astype(np.int64)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    acc       = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    out = {
        "precision": float(precision),
        "recall":    float(recall),
        "f1":        float(f1),
        "accuracy":  float(acc),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }

    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        if len(np.unique(y_true)) == 2:
            out["auc"]    = float(roc_auc_score(y_true, y_prob))
            out["pr_auc"] = float(average_precision_score(y_true, y_prob))
        else:
            out["auc"]    = float("nan")
            out["pr_auc"] = float("nan")
    except Exception:
        out["auc"]    = float("nan")
        out["pr_auc"] = float("nan")

    return out


def get_recipe_id(video_id: str) -> str:
    """Estrae il recipe_id (es. '1') dal video_id (es. '1_7')."""
    return video_id.split("_")[0]


class VideoDataset:
    """Carica tutti i file .npz dalla cartella features_dir."""
    def __init__(self, features_dir: Path):
        self.video_embeddings = []
        for p in sorted(features_dir.glob("*.npz")):
            video_id = p.stem
            v = np.load(p)
            embedding = v["step_embedding"]  # shape (num_steps, embedding_dim)
            recipe = get_recipe_id(video_id)
            self.video_embeddings.append((video_id, recipe, embedding))


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, 5000, model_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(model_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_key_padding_mask):
        x = self.input_projection(x)
        x = x + self.pos_encoder[:, :x.size(1), :]
        x = self.dropout(x)

        output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        mask = src_key_padding_mask.unsqueeze(-1).float()
        output = output * (1 - mask)
        sum_output   = output.sum(dim=1)
        count_output = (1 - mask).sum(dim=1).clamp(min=1e-9)
        pooled = sum_output / count_output

        return self.classifier(pooled).view(-1)


class RecipeDataset(Dataset):
    def __init__(self, data_list, label_dict):
        self.samples = []
        for v_id, _, embedding in data_list:
            label = label_dict[v_id]
            self.samples.append((
                torch.tensor(embedding, dtype=torch.float32),
                torch.tensor(label,     dtype=torch.long),
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    embeddings, labels = zip(*batch)
    embeddings_padded = pad_sequence(embeddings, batch_first=True)
    mask = torch.zeros(embeddings_padded.shape[:2], dtype=torch.bool)
    for i, emb in enumerate(embeddings):
        mask[i, len(emb):] = True
    return embeddings_padded, torch.stack(list(labels)), mask


def build_optimizer(model: nn.Module, hparams: dict) -> torch.optim.Optimizer:
    name = hparams.get("optimizer", "adamw").lower()
    lr   = hparams["lr"]

    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=hparams.get("weight_decay", 1e-2),
        )
    elif name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Ottimizzatore non supportato: '{name}'. Scegli tra 'adam' e 'adamw'.")


def Transform_fold(train_data, test_data, label_dict, hparams, fold_idx: int, use_wandb: bool):

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = RecipeDataset(train_data, label_dict)
    test_ds  = RecipeDataset(test_data,  label_dict)

    train_loader = DataLoader(train_ds, batch_size=hparams['batch_size'], shuffle=True,  collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = TransformerClassifier(
        input_dim  = hparams['input_dim'],
        model_dim  = hparams['model_dim'],
        num_heads  = hparams['num_heads'],
        num_layers = hparams['num_layers'],
        dropout    = hparams['dropout'],
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = build_optimizer(model, hparams)

    # --- Training ---
    model.train()
    for epoch in range(hparams['epochs']):
        epoch_loss = 0.0
        n_batches  = 0
        for batch_x, batch_y, mask in train_loader:
            batch_x, batch_y, mask = batch_x.to(device), batch_y.to(device), mask.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x, mask)
            loss = criterion(outputs, batch_y.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                f"fold_{fold_idx}/train_loss": avg_loss,
                f"fold_{fold_idx}/epoch":      epoch,
            })

    # --- Evaluation ---
    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for batch_x, batch_y, mask in test_loader:
            batch_x, mask = batch_x.to(device), mask.to(device)
            logits = model(batch_x, mask)
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_targets.extend(batch_y.numpy())

    test_video_ids = [v_id for v_id, _, _ in test_data]

    return np.array(all_targets), np.array(all_probs), test_video_ids


def save_predictions(output_path: str, fold_results: list, overall_metrics: dict):
    output = {
        "folds": fold_results,
        "overall_metrics": overall_metrics,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nPredizioni salvate in: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="LOO Transformer Classifier")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="FILE",
        help="Percorso del file JSON in cui salvare le predizioni (es. predictions.json).",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adam", "adamw"],
        help="Ottimizzatore da usare: 'adam' oppure 'adamw' (default).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-2,
        dest="weight_decay",
        help="Weight decay per AdamW (default: 0.01). Ignorato se si usa Adam.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        dest="use_wandb",
        help="Abilita il logging su Weights & Biases.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="loo-transformer",
        dest="wandb_project",
        help="Nome del progetto W&B (default: 'loo-transformer').",
    )
    parser.add_argument(
        "--wandb-run",
        type=str,
        default=None,
        dest="wandb_run",
        help="Nome della run W&B. Se non specificato, W&B ne genera uno automatico.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.use_wandb and not WANDB_AVAILABLE:
        raise ImportError("wandb non è installato. Esegui: pip install wandb")
    #KFold
    #features_dir = Path("output_KFold__step_embedding")
    #KFold 1s
    features_dir = Path("output_KFold_1s_step_embedding")
    #Normal
    #features_dir = Path("output_step_embeddings")
    label_csv    = Path("annotations/annotation_csv/step_annotations.csv")

    label_dict = CSV_read(label_csv)
    dataset    = VideoDataset(features_dir).video_embeddings

    unique_recipes = sorted(set(recipe for _, recipe, _ in dataset))
    n_recipes = len(unique_recipes)

    hyperparameters = {
       'input_dim':    1024,
        'model_dim':    256,
        'num_heads':    4,
        'num_layers':   1,
        'dropout':      0.1,
        'lr':           2e-4,
        'epochs':       10,
        'batch_size':   4,
        'optimizer':    args.optimizer,
        'weight_decay': args.weight_decay,
    }



    print(f"Ottimizzatore: {args.optimizer.upper()}"
          + (f"  weight_decay={args.weight_decay}" if args.optimizer == "adamw" else ""))

    # --- Inizializzazione W&B ---
    if args.use_wandb:
        wandb.init(
            project = args.wandb_project,
            name    = args.wandb_run,
            config  = hyperparameters,   # salva tutti gli iperparametri → confrontabili tra run
        )
        print(f"W&B run: {wandb.run.url}")

    all_y_true, all_y_prob = [], []
    fold_results = []

    for i, recipe in enumerate(unique_recipes):
        print(f"LOO fold {i+1}/{n_recipes} — recipe '{recipe}'")

        train_data = [(v, r, e) for v, r, e in dataset if r != recipe]
        test_data  = [(v, r, e) for v, r, e in dataset if r == recipe]

        if len(train_data) == 0 or len(test_data) == 0:
            print(f"  Skipped: train={len(train_data)}, test={len(test_data)}")
            continue

        y_true, y_prob, test_video_ids = Transform_fold(
            train_data, test_data, label_dict, hyperparameters,
            fold_idx=i + 1, use_wandb=args.use_wandb,
        )
        metrics = binary_metrics(y_true, y_prob)
        print(f"  Metrics: {metrics}")

        # Log metriche per fold su W&B
        if args.use_wandb:
            wandb.log({
                f"fold_{i+1}/precision": metrics["precision"],
                f"fold_{i+1}/recall":    metrics["recall"],
                f"fold_{i+1}/f1":        metrics["f1"],
                f"fold_{i+1}/accuracy":  metrics["accuracy"],
                f"fold_{i+1}/auc":       metrics.get("auc", float("nan")),
            })

        all_y_true.extend(y_true)
        all_y_prob.extend(y_prob)

        if args.output:
            y_pred = (y_prob >= 0.5).astype(int)
            fold_results.append({
                "fold":    i + 1,
                "recipe":  recipe,
                "metrics": metrics,
                "predictions": [
                    {
                        "video_id": vid,
                        "y_true":   int(yt),
                        "y_prob":   float(yp),
                        "y_pred":   int(ypr),
                    }
                    for vid, yt, yp, ypr in zip(test_video_ids, y_true, y_prob, y_pred)
                ],
            })

    # Metriche aggregate
    overall = {}
    if all_y_true:
        overall = binary_metrics(np.array(all_y_true), np.array(all_y_prob))
        print(f"\n=== Overall LOO metrics ===\n{overall}")

        # Log metriche overall su W&B (visibili nel summary della run → facili da confrontare)
        if args.use_wandb:
            wandb.log({
                "overall/precision": overall["precision"],
                "overall/recall":    overall["recall"],
                "overall/f1":        overall["f1"],
                "overall/accuracy":  overall["accuracy"],
                "overall/auc":       overall.get("auc", float("nan")),
                "overall/pr_auc":    overall.get("pr_auc", float("nan")),
            })
            # Scrivi anche nel summary così appaiono nella tabella dei run W&B
            for k, v in overall.items():
                wandb.run.summary[f"overall/{k}"] = v

    if args.output:
        save_predictions(args.output, fold_results, overall)

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()