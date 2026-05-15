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
    group_id = wandb.util.generate_id()
    
    all_vids = []
    all_predictions = []
    all_ground_truths = []
    all_probs = [] #auroc

    print(f"Starting Leave-One-Out on {len(video_ids)} videos...")

    for fold, (train_idx, test_idx) in enumerate(loo.split(video_ids)):
        train_vids, test_vids = video_ids[train_idx], video_ids[test_idx]
        current_video = test_vids[0]
        
        run = wandb.init(
            project="Mistake-Detection-LOO-Final",
            group=group_id,
            name=f"fold_{fold}_{current_video}",
            reinit=True,
            config={
                "learning_rate": 1e-3,
                "dropout": 0.3,
                "embed_dim": 128,
                "num_layers": 4,
                "num_heads": 4,
                "batch_size": 16,
                "epochs": 20
            }
        )
        c = wandb.config

        train_ds = TaskVerificationDataset(npz_path, annotations_path, train_vids, split='train')
        test_ds = TaskVerificationDataset(npz_path, annotations_path, test_vids, split='test')
        train_loader = DataLoader(train_ds, batch_size=c.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=1)

        model = TaskVerificationTransformer(
            input_dim=768, embed_dim=c.embed_dim, num_layers=c.num_layers, num_heads=c.num_heads, dropout=c.dropout
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=c.learning_rate, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        # training
        for epoch in range(c.epochs):
            model.train()
            for batch in train_loader:
                feats, labels, masks = batch['features'].to(device), batch['label'].to(device).float(), batch['mask'].to(device)
                optimizer.zero_grad()
                loss = criterion(model(feats, masks), labels)
                loss.backward()
                optimizer.step()

        # evaluation
        model.eval()
        with torch.no_grad():
            batch = next(iter(test_loader))
            logits = model(batch['features'].to(device), batch['mask'].to(device))
            
            prob = torch.sigmoid(logits).item() #auroc
            pred = 1 if prob > 0.5 else 0
            gt = batch['label'].item()
            
            all_vids.append(current_video)
            all_predictions.append(pred)
            all_ground_truths.append(gt)
            all_probs.append(prob)
            
        wandb.log({"fold_correct": int(pred == gt), "video_id": current_video})
        run.finish()

        if fold % 10 == 0:
            print(f"Completed fold {fold}/{len(video_ids)}...")

    # --- FINAL METRICS ---
    acc = accuracy_score(all_ground_truths, all_predictions)
    prec = precision_score(all_ground_truths, all_predictions)
    rec = recall_score(all_ground_truths, all_predictions)
    f1 = f1_score(all_ground_truths, all_predictions)
    auroc = roc_auc_score(all_ground_truths, all_probs)

    print(f"\n--- FINAL RESULTS LOO ---")
    print(f"Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUROC: {auroc:.4f}")

    # --- Error Analysis ---
    #csv
    results_df = pd.DataFrame({
        'video_id': all_vids,
        'ground_truth': all_ground_truths,
        'prediction': all_predictions,
        'probability': all_probs,
        'is_correct': np.array(all_predictions) == np.array(all_ground_truths)
    })
    results_df.to_csv('loo_error_analysis.csv', index=False)

    # wandb table
    final_run = wandb.init(project="Mistake-Detection-LOO-Final", name="Final_Metrics_Summary")
    error_table = wandb.Table(dataframe=results_df)
    wandb.log({
        "Final_Accuracy": acc,
        "Final_Precision": prec,
        "Final_Recall": rec,
        "Final_F1": f1,
        "Final_AUROC": auroc,
        "Error_Analysis_Table": error_table
    })
    final_run.finish()

if __name__ == "__main__":
    train_loo('step_embeddings_dataset.npz', 'annotations/annotation_json/complete_step_annotations.json')