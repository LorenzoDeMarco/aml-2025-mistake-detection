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


def train_loo(fold_id, train_ids, test_ids, global_visual, global_text, global_matches, args):
    
    wandb.init(
        project="aml-mistake-detection-gnn",
        name=f"LOO_Fold_{fold_id}",
        config=args,
        reinit=True,
        mode="online"
    )
    
    #graph-aware datasets using preloaded RAM features
    train_dataset = TaskVerificationGraphDataset(
        preloaded_visual=global_visual,
        preloaded_text=global_text,
        preloaded_matches=global_matches,
        graph_zip_path=args['graph_zip'],
        annotations_path=args['annotations_json'],
        video_ids=train_ids,
        split='train'
    )
    
    test_dataset = TaskVerificationGraphDataset(
        preloaded_visual=global_visual,
        preloaded_text=global_text,
        preloaded_matches=global_matches,
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
    
    #symmetric binary cross entropy with label smoothing 
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
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
            precomputed_matches = batch["precomputed_matches"]
            labels = batch["labels"].to(device)
            
            # label smoothing -> [0.1, 0.9]
            smoothed_labels = labels * (1.0 - label_smoothing) + label_smoothing / 2.0
            
            optimizer.zero_grad()
            logits, align_loss = model(vis_feat, text_feat, vis_mask, text_mask, edge_idx_list, precomputed_matches)
            
            classification_loss = criterion(logits, smoothed_labels)
            total_loss = classification_loss + 0.1 * align_loss
            
            total_loss.backward()
            optimizer.step()
            train_loss += classification_loss.item() * vis_feat.size(0)
            
        scheduler.step()
        epoch_loss = train_loss / len(train_dataset)
        wandb.log({"train/loss": epoch_loss, "train/lr": scheduler.get_last_lr()[0], "epoch": epoch})
        
    #validation (Leave-One-Out Evaluation)
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
            precomputed_matches = batch["precomputed_matches"]
            labels = batch["labels"]
            
            logits, _ = model(vis_feat, text_feat, vis_mask, text_mask, edge_idx_list, precomputed_matches)
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
        'matches_npz': 'hungarian_matches.npz',
        'graph_zip': 'task_graphs',
        'annotations_json': 'annotations/annotation_json/complete_step_annotations.json',
        'batch_size': 64,
        'epochs': 20,
        'lr': 2e-4,
        'weight_decay': 1e-2,
        'dropout': 0.4
    }
    
   
    print("Executing RAM caching ...")
    global_visual = {}
    with np.load(hyperparameters['visual_npz']) as data:
        for k in data.files:
            global_visual[k.replace('.npy', '')] = data[k].astype(np.float32)
            
    global_text = {}
    with np.load(hyperparameters['text_npz']) as data:
        for k in data.files:
            global_text[k.replace('.npy', '')] = data[k].astype(np.float32)
            
    global_matches = {}
    if not os.path.exists(hyperparameters['matches_npz']):
        raise FileNotFoundError(f"Please run 'python -m task_verification.precompute_matching' once to generate {hyperparameters['matches_npz']} before training.")
        
    with np.load(hyperparameters['matches_npz']) as data:
        for k in data.files:
            global_matches[k.replace('.npy', '')] = data[k]
            
    #load all unique video keys to perform the Leave-One-Out Cross Validation
    all_video_ids = sorted(list(global_visual.keys()))
    
    progress_records = []
    print(f"Starting Leave-One-Out Validation across {len(all_video_ids)} folds on A100 node.")
    
    for fold, test_id in enumerate(all_video_ids):
        train_ids = [vid for vid in all_video_ids if vid != test_id]
        test_ids = [test_id]
        
        vid_key, gt, pred_prob = train_loo(fold, train_ids, test_ids, global_visual, global_text, global_matches, hyperparameters)
        pred_class = 1 if pred_prob >= 0.5 else 0
        
        progress_records.append({
            "video_id": vid_key,
            "ground_truth": int(gt),
            "prediction": pred_class,
            "probability": float(pred_prob)
        })
        print(f"Fold {fold+1}/{len(all_video_ids)} | Video: {vid_key} | GT: {gt} | Prob: {pred_prob:.4f}")
        
        #iterative checkpoint 
        if (fold + 1) % 5 == 0 or (fold + 1) == len(all_video_ids):
            df_progress = pd.DataFrame(progress_records)
            df_progress.to_csv("loo_gnn_error_analysis.csv", index=False)
            
        #final global performance
        if (fold + 1) == len(all_video_ids):
            df_progress = pd.DataFrame(progress_records)
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
            print(f"wandz:   Final_Acc {final_acc:.5f}")
            print(f"wandb:    Final_F1 {final_f1:.5f}")
            print(f"wandb:  Final_Prec {final_prec:.5f}")
            print(f"wandb:   Final_Rec {final_rec:.5f}")
            print("=========================================================================\n")