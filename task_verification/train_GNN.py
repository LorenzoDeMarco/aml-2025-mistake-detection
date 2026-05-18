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

from task_verification.dataset_GNN import TaskVerificationGraphDataset, graph_collate_fn
from task_verification.GNN import TaskVerificationGNN

def train_loo(fold_id, train_ids, test_ids, args):
    wandb.init(
        project="aml-mistake-detection-gnn",
        name=f"LOO_Fold_{fold_id}",
        config=args,
        reinit=True,
        mode="online"
    )
    
    #graph-aware datasets
    train_dataset = TaskVerificationGraphDataset(
        visual_npz_path=args['visual_npz'],
        text_npz_path=args['text_npz'],
        graph_zip_path=args['graph_zip'],
        annotations_path=args['annotations_json'],
        video_ids=train_ids,
        split='train'
    )
    
    test_dataset = TaskVerificationGraphDataset(
        visual_npz_path=args['visual_npz'],
        text_npz_path=args['text_npz'],
        graph_zip_path=args['graph_zip'],
        annotations_path=args['annotations_json'],
        video_ids=test_ids,
        split='test'
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
    
    optimizer = optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['epochs'], eta_min=1e-6)
    
    #BCE with 0.1 Label Smoothing factor to regularize logit distributions
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.2], device=device), reduction='mean')
    label_smoothing = 0.1

    #training loop
    for epoch in range(1, args['epochs'] + 1):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            vis_feat = batch["visual_features"].to(device)
            text_feat = batch["text_features"].to(device)
            vis_mask = batch["visual_mask"].to(device)
            text_mask = batch["text_mask"].to(device)
            edge_idx_list = batch["edge_indices"] 
            labels = batch["labels"].to(device)
            
            #amooth binary targets: 1 -> 0.9, 0 -> 0.1
            smoothed_labels = labels * (1.0 - label_smoothing) + 0.5 * label_smoothing
            
            optimizer.zero_grad()
            logits = model(vis_feat, text_feat, vis_mask, text_mask, edge_idx_list)
            loss = criterion(logits, smoothed_labels)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * vis_feat.size(0)
            
        scheduler.step()
        epoch_loss = train_loss / len(train_dataset)
        wandb.log({"train/loss": epoch_loss, "train/lr": scheduler.get_last_lr()[0], "epoch": epoch})
        
    # validatin
    model.eval()
    test_probs = []
    test_gts = []
    video_keys = []
    
    with torch.no_grad():
        for batch in test_loader:
            vis_feat = batch["visual_features"].to(device)
            text_feat = batch["text_features"].to(device)
            vis_mask = batch["visual_mask"].to(device)
            text_mask = batch["text_mask"].to(device)
            edge_idx_list = batch["edge_indices"]
            labels = batch["labels"]
            
            logits = model(vis_feat, text_feat, vis_mask, text_mask, edge_idx_list)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            test_probs.extend(probs)
            test_gts.extend(labels.numpy())
            video_keys.extend(batch["video_ids"])
            
    pred_class = 1 if test_probs[0] >= 0.5 else 0
    wandb.log({
        "test/ground_truth": int(test_gts[0]),
        "test/probability": float(test_probs[0]),
        "test/prediction": pred_class,
        "test/is_correct": int(pred_class == int(test_gts[0]))
    })
            
    wandb.finish()
    return video_keys[0], test_gts[0], test_probs[0]

if __name__ == "__main__":
    

    hyperparameters = {
        'visual_npz': 'step_embeddings_dataset.npz',
        'text_npz': 'text_task_graphs.npz',
        'graph_zip': 'annotations/task_graphs',
        'annotations_json': 'annotations/annotation_json/complete_step_annotations.json',
        'batch_size': 64,
        'epochs': 20,
        'lr': 2e-4,
        'weight_decay': 1e-2,
        'dropout': 0.4
    }
    
    all_features = np.load(hyperparameters['visual_npz'])
    all_video_ids = sorted(list(all_features.keys()))
    
    progress_records = []
    print(f"Starting Leave-One-Out Validation across {len(all_video_ids)} folds on A100 node.")
    
    for fold, test_id in enumerate(all_video_ids):
        train_ids = [vid for vid in all_video_ids if vid != test_id]
        test_ids = [test_id]
        
        vid_key, gt, pred_prob = train_loo(fold, train_ids, test_ids, hyperparameters)
        pred_class = 1 if pred_prob >= 0.5 else 0
        
        progress_records.append({
            "video_id": vid_key,
            "ground_truth": int(gt),
            "prediction": pred_class,
            "probability": float(pred_prob)
        })
        print(f"Fold {fold+1}/{len(all_video_ids)} | Video: {vid_key} | GT: {gt} | Prob: {pred_prob:.4f}")
        
        if (fold + 1) % 5 == 0 or (fold + 1) == len(all_video_ids):
            df_progress = pd.DataFrame(progress_records)
            df_progress.to_csv("loo_gnn_error_analysis.csv", index=False)
            
        #final summary
        if (fold + 1) == len(all_video_ids):
            y_true = df_progress['ground_truth'].values
            y_pred = df_progress['prediction'].values
            y_prob = df_progress['probability'].values
            
            final_acc = accuracy_score(y_true, y_pred)
            final_prec = precision_score(y_true, y_pred, zero_division=0)
            final_rec = recall_score(y_true, y_pred, zero_division=0)
            final_f1 = f1_score(y_true, y_pred, zero_division=0)
            final_auroc = roc_auc_score(y_true, y_prob)
            
            print("\n================ FINAL GLOBAL GNN PERFORMANCE EVALUATION ================")
            print(f"wandb: Final_AUROC {final_auroc:.5f}")
            print(f"wandb:   Final_Acc {final_acc:.5f}")
            print(f"wandb:    Final_F1 {final_f1:.5f}")
            print(f"wandb:  Final_Prec {final_prec:.5f}")
            print(f"wandb:   Final_Rec {final_rec:.5f}")
            print("=========================================================================\n")