import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import wandb
import random
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

from task_verification.dataset import TaskVerificationDataset
from task_verification.transformer import TaskVerificationTransformer


def dynamic_collate_fn(batch):
    features = [item['features'] for item in batch]
    labels = [item['label'] for item in batch]
    video_ids = [item['video_id'] for item in batch]

    padded_features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0.0)

    batch_size, max_len, _ = padded_features.shape
    masks = torch.zeros((batch_size, max_len), dtype=torch.float32)

    for i, feat in enumerate(features):
        actual_len = feat.shape[0]
        masks[i, :actual_len] = 1.0

    return {
        'features': padded_features,
        'label': torch.tensor(labels, dtype=torch.long),
        'mask': masks,
        'video_id': video_ids
    }


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    set_seed(42)

    npz_path = 'step_embeddings_dataset.npz'
    annotations_path = 'annotations/annotation_json/complete_step_annotations.json'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = torch.cuda.is_available()
    print(f"Using device: {device} | AMP: {use_amp}")

    # AMP scaler with CPU fallback
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    all_vids = sorted(list(annotations.keys()))

    # --- LOGO Grouping Logic ---
    recipe_groups = {}
    for vid in all_vids:
        recipe_id = vid.split('_')[0]
        if recipe_id not in recipe_groups:
            recipe_groups[recipe_id] = []
        recipe_groups[recipe_id].append(vid)

    # --- Checkpoint / resume logic ---
    progress_file = 'logo_progress.csv'
    if os.path.exists(progress_file):
        print(f"Found existing progress file '{progress_file}'. Resuming training...", flush=True)
        df_progress = pd.read_csv(progress_file)
        progress_records = df_progress.to_dict('records')
        completed_recipes = set(df_progress['recipe_id'].tolist()) if 'recipe_id' in df_progress.columns else set()
        all_ground_truths = df_progress['ground_truth'].tolist()
        all_predictions = df_progress['prediction'].tolist()
        all_probs = df_progress['probability'].tolist()
        group_id = df_progress['group_id'].iloc[0] if 'group_id' in df_progress.columns else wandb.util.generate_id()
    else:
        print("No progress file found. Starting a LOGO run...", flush=True)
        progress_records = []
        completed_recipes = set()
        all_ground_truths = []
        all_predictions = []
        all_probs = []
        group_id = wandb.util.generate_id()

    print(f"\n=======================================================================")
    print(f" Starting Leave-One-Group-Out (LOGO) across {len(recipe_groups)} recipes.")
    print(f"=======================================================================\n")

    global_start_time = time.time()

    for fold, (recipe_id, test_vids) in enumerate(recipe_groups.items()):

        if recipe_id in completed_recipes:
            print(f"[{fold+1}/{len(recipe_groups)}] Skipping already completed recipe: '{recipe_id}'", flush=True)
            continue

        fold_start_time = time.time()
        train_vids = [vid for vid in all_vids if vid not in test_vids]

        print(f"\n[{fold+1}/{len(recipe_groups)}] Start Fold | Testing unseen recipe: '{recipe_id}' ({len(test_vids)} videos)")

        run = wandb.init(
            project="Mistake-Detection-LOGO",
            group=group_id,
            name=f"LOGO_Recipe_{recipe_id}",
            reinit=True,
            config={
                "learning_rate": 2e-4,
                "dropout": 0.4,
                "embed_dim": 256,
                "num_layers": 2,
                "num_heads": 8,
                "batch_size": 64,
                "epochs": 20
            }
        )
        c = wandb.config

        train_dataset = TaskVerificationDataset(npz_path, annotations_path, train_vids, split='train')
        test_dataset = TaskVerificationDataset(npz_path, annotations_path, test_vids, split='test')

        train_loader = DataLoader(train_dataset, batch_size=c.batch_size, shuffle=True, collate_fn=dynamic_collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=c.batch_size, shuffle=False, collate_fn=dynamic_collate_fn)

        model = TaskVerificationTransformer(
            input_dim=768, embed_dim=c.embed_dim, num_layers=c.num_layers,
            num_heads=c.num_heads, dropout=c.dropout, max_seq_len=1050
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=c.learning_rate, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=c.epochs, eta_min=1e-6)
        criterion = nn.BCEWithLogitsLoss()

        # --- Training Loop ---
        label_smoothing = 0.1
        for epoch in range(c.epochs):
            model.train()
            epoch_loss = 0.0

            for batch in train_loader:
                feats = batch['features'].to(device)
                labels = batch['label'].to(device).float()
                masks = batch['mask'].to(device)

                # Label smoothing [0.1, 0.9] to prevent overconfidence
                smoothed_labels = labels * (1.0 - label_smoothing) + (1.0 - labels) * label_smoothing
                optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(feats, masks)
                    loss = criterion(logits, smoothed_labels)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()

            scheduler.step()

            avg_loss = epoch_loss / len(train_loader)
            wandb.log({"epoch/train_loss": avg_loss, "epoch/epoch_step": epoch})

            if epoch == c.epochs - 1:
                print(f"  [Fold {fold+1}] Final Loss: {avg_loss:.4f}", flush=True)

        # --- Test Evaluation ---
        model.eval()
        fold_ground_truths = []
        fold_predictions = []
        fold_probs = []
        fold_video_ids = []

        with torch.no_grad():
            for batch in test_loader:
                feats = batch['features'].to(device)
                labels = batch['label'].to(device)
                masks = batch['mask'].to(device)
                v_ids = batch['video_id']

                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(feats, masks)

                probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
                preds = (probs >= 0.5).astype(int)

                if probs.ndim == 0:
                    probs = np.expand_dims(probs, axis=0)
                    preds = np.expand_dims(preds, axis=0)

                fold_ground_truths.extend(labels.cpu().numpy().tolist())
                fold_predictions.extend(preds.tolist())
                fold_probs.extend(probs.tolist())
                fold_video_ids.extend(v_ids)

        # Aggregate globally
        all_ground_truths.extend(fold_ground_truths)
        all_predictions.extend(fold_predictions)
        all_probs.extend(fold_probs)

        for vid, gt, pred, prob in zip(fold_video_ids, fold_ground_truths, fold_predictions, fold_probs):
            progress_records.append({
                "recipe_id": recipe_id,
                "video_id": vid,
                "ground_truth": gt,
                "prediction": pred,
                "probability": prob,
                "group_id": group_id
            })

        elapsed_fold_time = time.time() - fold_start_time
        running_acc = accuracy_score(all_ground_truths, all_predictions)

        wandb.log({
            "fold_running_accuracy": running_acc,
            "fold_time_sec": elapsed_fold_time,
            "recipe_id": recipe_id
        })
        run.finish()

        # Save checkpoint
        df_progress = pd.DataFrame(progress_records)
        df_progress.to_csv(progress_file, index=False)

        print(f"    -> Done | Running Global Acc: {running_acc:.4f} | Time: {elapsed_fold_time:.2f}s", flush=True)

    # --- Global Metrics ---
    acc = accuracy_score(all_ground_truths, all_predictions)
    prec = precision_score(all_ground_truths, all_predictions)
    rec = recall_score(all_ground_truths, all_predictions)
    f1 = f1_score(all_ground_truths, all_predictions)
    auroc = roc_auc_score(all_ground_truths, all_probs)

    print(f"\n--- FINAL LOGO RESULTS ---")
    print(f"Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUROC: {auroc:.4f}")

    results_df = pd.DataFrame(progress_records)
    results_df['is_correct'] = results_df['prediction'] == results_df['ground_truth']
    results_df.to_csv('logo_error_analysis_final.csv', index=False)

    final_run = wandb.init(project="Mistake-Detection-LOGO", name="GLOBAL_METRICS_SUMMARY")
    error_table = wandb.Table(dataframe=results_df)

    wandb.log({
        "Final_Acc": acc,
        "Final_F1": f1,
        "Final_Prec": prec,
        "Final_Rec": rec,
        "Final_AUROC": auroc,
        "conf_mat": wandb.plot.confusion_matrix(
            probs=None,
            y_true=all_ground_truths,
            preds=all_predictions,
            class_names=["Correct", "Error"]
        ),
        "error_analysis_table": error_table
    })
    final_run.finish()

    global_duration = time.time() - global_start_time
    print(f"\nExecution completed in {global_duration / 60:.2f} minutes.")


if __name__ == "__main__":
    main()
