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
from sklearn.model_selection import LeaveOneOut
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

def train_loo(npz_path, annotations_path):
    set_seed(42) 
    
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    video_ids = np.array(list(annotations.keys()))
    
    loo = LeaveOneOut()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- checkpoint with resume logic ---
    progress_file = 'loo_progress.csv'
    if os.path.exists(progress_file):
        print(f"Found existing progress file '{progress_file}'. Resuming training...", flush=True)
        df_progress = pd.read_csv(progress_file)
        all_vids = df_progress['video_id'].tolist()
        all_predictions = df_progress['prediction'].tolist()
        all_ground_truths = df_progress['ground_truth'].tolist()
        all_probs = df_progress['probability'].tolist()
        completed_videos = set(all_vids)
        group_id = df_progress['group_id'].iloc[0] if 'group_id' in df_progress.columns else wandb.util.generate_id()
    else:
        print("No progress file found. Starting a LOO run...", flush=True)
        all_vids, all_predictions, all_ground_truths, all_probs = [], [], [], []
        completed_videos = set()
        group_id = wandb.util.generate_id()

    #AMP Scaler initialization for high-speed float16 execution
    scaler = torch.cuda.amp.GradScaler()

    print(f"Starting Leave-One-Out on {len(video_ids)} videos...", flush=True)

    for fold, (train_idx, test_idx) in enumerate(loo.split(video_ids)):
        train_vids, test_vids = video_ids[train_idx], video_ids[test_idx]
        current_video = test_vids[0]
        
        if current_video in completed_videos:
            continue
            
        fold_start_time = time.time()
        
        run = wandb.init(
            project="Mistake-Detection-LOO",
            group=group_id,
            name=f"fold_{fold}_{current_video}",
            reinit=True,
            config={
                "learning_rate": 2e-4, 
                "dropout": 0.2,
                "embed_dim": 256,       
                "num_layers": 2,
                "num_heads": 8,
                "batch_size": 64,     
                "epochs": 12        
            }
        )
        c = wandb.config

        train_ds = TaskVerificationDataset(npz_path, annotations_path, train_vids, split='train')
        test_ds = TaskVerificationDataset(npz_path, annotations_path, test_vids, split='test')
        train_loader = DataLoader(train_ds, batch_size=c.batch_size, shuffle=True, collate_fn=dynamic_collate_fn)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=dynamic_collate_fn)

        model = TaskVerificationTransformer(
            input_dim=768, embed_dim=c.embed_dim, num_layers=c.num_layers, num_heads=c.num_heads, dropout=c.dropout, max_seq_len=1050
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=c.learning_rate, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=c.epochs, eta_min=1e-6)

        #loss setup with class imbalance handling
        #pos_weight_val = 164.0 / 220.0
        ##criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]).to(device))

        #symmetric loss performs better with label smoothing and prevents overconfidence on the minority class, improving generalization
        criterion = nn.BCEWithLogitsLoss()

        # training loop with mixed precision and label smoothing
        label_smoothing = 0.1
        for epoch in range(c.epochs):
            model.train()
            epoch_loss = 0.0
            
            for batch in train_loader:
                feats = batch['features'].to(device)
                labels = batch['label'].to(device).float()
                masks = batch['mask'].to(device)
                
                # label smoothin [0.1, 0.9] to prevent overconfidence and improve generalization
                smoothed_labels = labels * (1.0 - label_smoothing) + (1.0 - labels) * label_smoothing
                optimizer.zero_grad()
                
                #forwardexecution enclosed inside AMP autocast
                with torch.cuda.amp.autocast():
                    logits = model(feats, masks)
                    loss = criterion(logits, smoothed_labels)
                
                scaler.scale(loss).backward()
                
                #before clippint to prevent Nan injections
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss += loss.item()
                
            scheduler.step()

            avg_loss = epoch_loss / len(train_loader)
            wandb.log({
                "epoch/train_loss": avg_loss,
                "epoch/epoch_step": epoch
            })

            if epoch == c.epochs - 1:
                print(f"  [Fold {fold}] Final Loss: {epoch_loss/len(train_loader):.4f}", flush=True)

        # evaluation
        model.eval()
        with torch.no_grad():
            batch = next(iter(test_loader))
            with torch.cuda.amp.autocast():
                logits = model(batch['features'].to(device), batch['mask'].to(device))
            
            prob = torch.sigmoid(logits).item() 
            pred = 1 if prob > 0.5 else 0
            gt = batch['label'].item()
            
            all_vids.append(current_video)
            all_predictions.append(pred)
            all_ground_truths.append(gt)
            all_probs.append(prob)
            
        elapsed_fold_time = time.time() - fold_start_time
        running_acc = accuracy_score(all_ground_truths, all_predictions)
        
        wandb.log({
            "fold_correct": int(pred == gt), 
            "video_id": current_video,
            "running_accuracy": running_acc,
            "fold_time_sec": elapsed_fold_time
        })
        run.finish()

        df_progress = pd.DataFrame({
            'video_id': all_vids,
            'ground_truth': all_ground_truths,
            'prediction': all_predictions,
            'probability': all_probs,
            'group_id': [group_id] * len(all_vids)
        })
        df_progress.to_csv(progress_file, index=False)

        if fold % 5 == 0:
            print(f"Fold {fold}/{len(video_ids)} Done | Running Global Acc: {running_acc:.4f} | Time: {elapsed_fold_time:.2f}s", flush=True)

    # --- global metrics ---
    acc = accuracy_score(all_ground_truths, all_predictions)
    prec = precision_score(all_ground_truths, all_predictions)
    rec = recall_score(all_ground_truths, all_predictions)
    f1 = f1_score(all_ground_truths, all_predictions)
    auroc = roc_auc_score(all_ground_truths, all_probs)

    print(f"\n--- FINAL RESULTS ---")
    print(f"Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUROC: {auroc:.4f}")

    results_df = pd.DataFrame({
        'video_id': all_vids,
        'ground_truth': all_ground_truths,
        'prediction': all_predictions,
        'probability': all_probs,
        'is_correct': np.array(all_predictions) == np.array(all_ground_truths)
    })
    results_df.to_csv('loo_error_analysis_final.csv', index=False)

    final_run = wandb.init(project="Mistake-Detection-LOO", name="GLOBAL_METRICS_SUMMARY")
    error_table = wandb.Table(dataframe=results_df)

    wandb.log({
        "Final_Acc": acc, "Final_F1": f1, "Final_Prec": prec, "Final_Rec": rec, "Final_AUROC": auroc,
        "conf_mat": wandb.plot.confusion_matrix(probs=None, y_true=all_ground_truths, preds=all_predictions, class_names=["Correct", "Error"]),
        "error_analysis_table": error_table
    })
    final_run.finish()

if __name__ == "__main__":
    train_loo('step_embeddings_dataset.npz', 'annotations/annotation_json/complete_step_annotations.json')