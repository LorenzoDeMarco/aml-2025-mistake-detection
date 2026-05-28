import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import wandb
import pandas as pd
import time

from task_verification.dataset_GNN import TaskVerificationGraphDataset, graph_collate_fn
from task_verification.GNN import TaskVerificationGNN


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_logo_fold(fold_id, recipe_id, train_ids, test_ids, global_visual, global_text, args):
    # Deterministic seed, different per fold to preserve cross-fold variability
    set_seed(args['base_seed'] + fold_id)

    wandb.init(
        project="aml-mistake-detection-gnn",
        name=f"LOGO_Recipe_{recipe_id}",
        config=args,
        reinit=True,
        mode="online"
    )

    train_dataset = TaskVerificationGraphDataset(
        preloaded_visual=global_visual, preloaded_text=global_text,
        graph_zip_path=args['graph_zip'], annotations_path=args['annotations_json'],
        video_ids=train_ids, split='train'
    )

    test_dataset = TaskVerificationGraphDataset(
        preloaded_visual=global_visual, preloaded_text=global_text,
        graph_zip_path=args['graph_zip'], annotations_path=args['annotations_json'],
        video_ids=test_ids, split='test'
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args['batch_size'], shuffle=True,
        num_workers=4, collate_fn=graph_collate_fn, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args['batch_size'], shuffle=False,
        num_workers=2, collate_fn=graph_collate_fn, pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TaskVerificationGNN(dropout=args['dropout']).to(device)

    projector_params, base_params = [], []
    for name, param in model.named_parameters():
        if 'sim_visual_proj' in name or 'sim_text_proj' in name or 'logit_scale' in name:
            projector_params.append(param)
        else:
            base_params.append(param)

    optimizer = optim.AdamW([
        {'params': base_params, 'lr': args['lr']},
        {'params': projector_params, 'lr': args['lr'] * 5.0}
    ], weight_decay=args['weight_decay'])

    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    label_smoothing = 0.1

    print(f"    [Setup] Train : {len(train_ids)} video | Test : {len(test_ids)} video | Seed: {args['base_seed'] + fold_id}")

    for epoch in range(1, args['epochs'] + 1):
        epoch_start_time = time.time()

        current_align_weight = max(0.1, 1.0 * (0.8 ** (epoch - 1)))

        model.train()
        train_loss = 0.0

        for batch in train_loader:
            vis_feat = batch["visual_features"].to(device)
            text_feat = batch["text_features"].to(device)
            vis_mask = batch["visual_mask"].to(device)
            text_mask = batch["text_mask"].to(device)
            edge_idx_list = batch["edge_indices"]
            node_depths = batch["node_depths"].to(device)
            labels = batch["labels"].to(device)

            smoothed_labels = labels * (1.0 - label_smoothing) + (1.0 - labels) * label_smoothing

            optimizer.zero_grad()
            logits, align_loss = model(vis_feat, text_feat, vis_mask, text_mask, edge_idx_list, node_depths)

            classification_loss = criterion(logits, smoothed_labels)
            total_loss = classification_loss + current_align_weight * align_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += classification_loss.item() * vis_feat.size(0)

        scheduler_lr = optimizer.param_groups[0]['lr']
        epoch_loss = train_loss / len(train_dataset)
        wandb.log({"train/loss": epoch_loss, "train/lr": scheduler_lr, "epoch": epoch, "align_weight": current_align_weight})

        epoch_duration = time.time() - epoch_start_time

        if epoch == 1 or epoch == args['epochs'] or epoch % 5 == 0:
            print(f"    -> Epoch {epoch:02d}/{args['epochs']} | Loss: {epoch_loss:.4f} | AlignWt: {current_align_weight:.3f} | Time: {epoch_duration:.2f}s")

    model.eval()
    fold_results = []

    with torch.no_grad():
        for batch in test_loader:
            vis_feat = batch["visual_features"].to(device)
            text_feat = batch["text_features"].to(device)
            vis_mask = batch["visual_mask"].to(device)
            text_mask = batch["text_mask"].to(device)
            edge_idx_list = batch["edge_indices"]
            node_depths = batch["node_depths"].to(device)
            labels = batch["labels"]
            video_ids = batch["video_ids"]

            logits, _ = model(vis_feat, text_feat, vis_mask, text_mask, edge_idx_list, node_depths)
            probs = torch.sigmoid(logits).cpu().numpy()
            gts = labels.numpy()

            for i in range(len(video_ids)):
                pred = 1 if probs[i] >= 0.5 else 0
                fold_results.append({
                    "video_id": video_ids[i],
                    "ground_truth": int(gts[i]),
                    "prediction": pred,
                    "probability": float(probs[i])
                })

    wandb.finish()
    return fold_results


if __name__ == "__main__":
    hyperparameters = {
        'visual_npz': 'step_embeddings_dataset.npz',
        'text_npz': 'text_task_graphs_v2.npz',
        'graph_zip': 'annotations/task_graphs',
        'annotations_json': 'annotations/annotation_json/complete_step_annotations.json',
        'batch_size': 16,
        'epochs': 45,
        'lr': 2e-4,
        'weight_decay': 1e-2,
        'dropout': 0.4,
        'base_seed': 42,
    }

    global_start_time = time.time()

    print("Executing RAM caching strategy...")
    global_visual = {k.replace('.npy', ''): v.astype(np.float32) for k, v in np.load(hyperparameters['visual_npz']).items()}
    global_text = {k.replace('.npy', ''): v.astype(np.float32) for k, v in np.load(hyperparameters['text_npz']).items()}

    all_video_ids = sorted(list(global_visual.keys()))

    recipe_groups = {}
    for vid in all_video_ids:
        recipe_id = vid.split('_')[0]
        if recipe_id not in recipe_groups:
            recipe_groups[recipe_id] = []
        recipe_groups[recipe_id].append(vid)

    progress_records = []
    print(f"\n=======================================================================")
    print(f" Starting Leave-One-Recipe-Out (LOGO) across {len(recipe_groups)} recipes.")
    print(f"=======================================================================\n")

    for fold, (recipe_id, test_ids) in enumerate(recipe_groups.items()):
        fold_start_time = time.time()

        train_ids = [vid for vid in all_video_ids if vid not in test_ids]

        print(f"\n[{fold+1}/{len(recipe_groups)}] Start Fold | Testing unseen recipe: '{recipe_id}' ({len(test_ids)} videos)")

        results = train_logo_fold(fold, recipe_id, train_ids, test_ids, global_visual, global_text, hyperparameters)

        print(f"    [Valutazione FOLD completata - Risultati sui {len(test_ids)} video unseen]:")
        for res in results:
            progress_records.append(res)
            print(f"      Video: {res['video_id']:<10} | GT: {res['ground_truth']} | Prob: {res['probability']:.4f}")

        df_progress = pd.DataFrame(progress_records)
        df_progress.to_csv("logo_gnn_error_analysis.csv", index=False)

        fold_duration = time.time() - fold_start_time
        print(f" Fold {fold+1} completed in {fold_duration:.2f} seconds ({fold_duration/60:.2f} minutes).")

    # Global metrics
    df_progress = pd.DataFrame(progress_records)
    y_true = df_progress['ground_truth'].values
    y_pred = df_progress['prediction'].values
    y_prob = df_progress['probability'].values

    final_acc = accuracy_score(y_true, y_pred)
    final_prec = precision_score(y_true, y_pred, zero_division=0)
    final_rec = recall_score(y_true, y_pred, zero_division=0)
    final_f1 = f1_score(y_true, y_pred, zero_division=0)
    final_auroc = roc_auc_score(y_true, y_prob)

    global_duration = time.time() - global_start_time

    print("\n================ FINAL GLOBAL GNN PERFORMANCE EVALUATION (LOGO) ================")
    print(f"wandb: Final_AUROC {final_auroc:.5f}")
    print(f"wandb:   Final_Acc {final_acc:.5f}")
    print(f"wandb:    Final_F1 {final_f1:.5f}")
    print(f"wandb:  Final_Prec {final_prec:.5f}")
    print(f"wandb:   Final_Rec {final_rec:.5f}")
    print("===============================================================================")
    print(f"Execution total completed in {global_duration/60:.2f} minutes ({(global_duration/3600):.2f} hours).\n")
