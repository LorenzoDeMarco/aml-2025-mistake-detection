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

    hyperparameters = {
        'visual_npz': 'step_embeddings.npz',
        'annotations_json': 'annotations/annotation_json/complete_step_annotations.json',
        'batch_size': 32,
        'epochs': 50,
        'lr': 1e-4,
        'weight_decay': 1e-3,
        'dropout': 0.3,
        'model_dim': 256,
        'num_heads': 4,
        'num_layers': 2
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading global datasets into RAM...")
    global_visual = {k.replace('.npy', ''): v for k, v in np.load(hyperparameters['visual_npz']).items()}

    all_vids = sorted(list(global_visual.keys()))

    # --- LOGO Grouping Logic ---
    recipe_groups = {}
    for vid in all_vids:
        recipe_id = vid.split('_')[0]
        if recipe_id not in recipe_groups:
            recipe_groups[recipe_id] = []
        recipe_groups[recipe_id].append(vid)

    progress_records = []
    all_ground_truths = []
    all_predictions = []
    all_probs = []

    print(f"\n=======================================================================")
    print(f" Starting Leave-One-Group-Out (LOGO) across {len(recipe_groups)} recipes.")
    print(f"=======================================================================\n")

    global_start_time = time.time()

    for fold, (recipe_id, test_vids) in enumerate(recipe_groups.items()):
        fold_start_time = time.time()
        
        train_vids = [vid for vid in all_vids if vid not in test_vids]

        print(f"\n[{fold+1}/{len(recipe_groups)}] Start Fold | Testing unseen recipe: '{recipe_id}' ({len(test_vids)} videos)")

        wandb.init(
            project="aml-mistake-detection-transformer-logo",
            name=f"LOGO_Recipe_{recipe_id}",
            config=hyperparameters,
            reinit=True,
            mode="online"
        )

        train_dataset = TaskVerificationDataset(
            preloaded_visual=global_visual,
            annotations_path=hyperparameters['annotations_json'],
            video_ids=train_vids,
            split='train'
        )

        test_dataset = TaskVerificationDataset(
            preloaded_visual=global_visual,
            annotations_path=hyperparameters['annotations_json'],
            video_ids=test_vids,
            split='test'
        )

        train_loader = DataLoader(train_dataset, batch_size=hyperparameters['batch_size'], shuffle=True, collate_fn=dynamic_collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=hyperparameters['batch_size'], shuffle=False, collate_fn=dynamic_collate_fn)

        model = TaskVerificationTransformer(
            input_dim=768,
            model_dim=hyperparameters['model_dim'],
            num_heads=hyperparameters['num_heads'],
            num_layers=hyperparameters['num_layers'],
            dropout=hyperparameters['dropout']
        ).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=hyperparameters['lr'], weight_decay=hyperparameters['weight_decay'])

        # --- Training Loop ---
        for epoch in range(hyperparameters['epochs']):
            model.train()
            train_loss = 0.0

            for batch in train_loader:
                v_feats = batch['features'].to(device)
                labels = batch['label'].to(device)
                masks = batch['mask'].to(device)

                optimizer.zero_grad()
                logits = model(v_feats, masks)
                loss = criterion(logits, labels.float().unsqueeze(1))
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * v_feats.size(0)

            avg_train_loss = train_loss / len(train_dataset)
            wandb.log({"train/loss": avg_train_loss, "epoch": epoch + 1})

        # --- Test Evaluation ---
        model.eval()
        test_loss = 0.0
        fold_ground_truths = []
        fold_predictions = []
        fold_probs = []
        fold_video_ids = []

        with torch.no_grad():
            for batch in test_loader:
                v_feats = batch['features'].to(device)
                labels = batch['label'].to(device)
                masks = batch['mask'].to(device)
                v_ids = batch['video_id']

                logits = model(v_feats, masks)
                loss = criterion(logits, labels.float().unsqueeze(1))
                test_loss += loss.item() * v_feats.size(0)

                probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
                preds = (probs >= 0.5).astype(int)

                if probs.ndim == 0:
                    probs = np.expand_dims(probs, axis=0)
                    preds = np.expand_dims(preds, axis=0)

                fold_ground_truths.extend(labels.cpu().numpy().tolist())
                fold_predictions.extend(preds.tolist())
                fold_probs.extend(probs.tolist())
                fold_video_ids.extend(v_ids)

        # Aggregate fold metrics globally
        all_ground_truths.extend(fold_ground_truths)
        all_predictions.extend(fold_predictions)
        all_probs.extend(fold_probs)

        # Record metrics for every video in the current fold
        for vid, gt, pred, prob in zip(fold_video_ids, fold_ground_truths, fold_predictions, fold_probs):
            progress_records.append({
                "video_id": vid,
                "ground_truth": gt,
                "prediction": pred,
                "probability": prob
            })

        # Calculate partial running metrics
        running_acc = accuracy_score(all_ground_truths, all_predictions)
        elapsed_fold_time = time.time() - fold_start_time

        wandb.finish()
        print(f"    -> Done | Running Global Acc: {running_acc:.4f} | Time: {elapsed_fold_time:.2f}s", flush=True)

    # --- Global Metrics ---
    acc = accuracy_score(all_ground_truths, all_predictions)
    prec = precision_score(all_ground_truths, all_predictions, zero_division=0)
    rec = recall_score(all_ground_truths, all_predictions, zero_division=0)
    f1 = f1_score(all_ground_truths, all_predictions, zero_division=0)
    auroc = roc_auc_score(all_ground_truths, all_probs)

    print(f"\n--- FINAL LOGO RESULTS ---")
    print(f"Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUROC: {auroc:.4f}")

    # Generate the DataFrame safely using the progress_records
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
        "Detailed_Results": error_table
    })
    wandb.finish()

    global_duration = time.time() - global_start_time
    print(f"\nExecution completed in {global_duration/60:.2f} minutes.")

if __name__ == "__main__":
    main()