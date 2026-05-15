from task_verification.dataset import TaskVerificationDataset
from task_verification.transformer import TaskVerificationTransformer
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import wandb
import math
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

def train_sweep():
    with wandb.init() as run:
        c = wandb.config
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #split recordings used in the  capitanCook paper
        with open('annotations/data_splits/recordings_data_split_combined.json', 'r') as f:
            split = json.load(f)

        train_ds = TaskVerificationDataset('step_embeddings_dataset.npz', 'annotations/annotation_json/complete_step_annotations.json', split['train'], split='train')
        val_ds = TaskVerificationDataset('step_embeddings_dataset.npz', 'annotations/annotation_json/complete_step_annotations.json', split['val'], split='val')
        
        train_loader = DataLoader(train_ds, batch_size=c.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=c.batch_size)

        model = TaskVerificationTransformer(embed_dim=c.embed_dim, num_heads=c.num_heads, num_layers=c.num_layers, dropout=c.dropout).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=c.learning_rate, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(30):
            model.train()
            for batch in train_loader:
                feats, labels, masks = batch['features'].to(device), batch['label'].to(device).float(), batch['mask'].to(device)
                optimizer.zero_grad()
                loss = criterion(model(feats, masks), labels)
                loss.backward()
                optimizer.step()

            model.eval()
            all_p, all_g = [], []
            with torch.no_grad():
                for batch in val_loader:
                    logits = model(batch['features'].to(device), batch['mask'].to(device))
                    preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
                    all_p.extend(preds); all_g.extend(batch['label'].numpy())
            
            val_f1 = f1_score(all_g, all_p)
            val_acc = accuracy_score(all_g, all_p)
            wandb.log({
                "epoch": epoch,
                "val_f1": val_f1,
                "val_acc": val_acc,
                "conf_mat": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=all_g, 
                    preds=all_p,
                    class_names=["Correct", "Error"]
                )
            })

sweep_config = {
    'method': 'random',
    'metric': {'name': 'val_f1', 'goal': 'maximize'},
    'parameters': {
        'learning_rate': {'values': [1e-3, 1e-4, 5e-5]},
        'dropout': {'values': [0.1, 0.3, 0.5]},
        'embed_dim': {'values': [128, 256]},
        'num_layers': {'values': [2, 4]},
        'num_heads': {'values': [4, 8]},
        'batch_size': {'values': [16, 32]}
    }
}

sweep_id = wandb.sweep(sweep_config, project="Mistake-Detection-Sweep")
wandb.agent(sweep_id, function=train_sweep, count=15)