##Leave-one-out strategy
from task_verification.dataset import TaskVerificationDataset
from task_verification.transformer import TaskVerificationTransformer
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
from sklearn.model_selection import LeaveOneOut
from torch.utils.data import DataLoader


def train_loo(npz_path, annotations_path):
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    video_ids = np.array(list(annotations.keys()))
    
    loo = LeaveOneOut()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    group_id = wandb.util.generate_id()
    results = []

    print(f"Starting LOO on {len(video_ids)} videos...")

    for fold, (train_idx, test_idx) in enumerate(loo.split(video_ids)):
        train_vids, test_vids = video_ids[train_idx], video_ids[test_idx]
        
        run = wandb.init(
            project="Mistake-Detection-LOO",
            group=group_id,
            name=f"fold_{fold}_{test_vids[0]}",
            reinit=True
        )

        # dataloader setup
        train_ds = TaskVerificationDataset(npz_path, annotations_path, train_vids, split='train')
        test_ds = TaskVerificationDataset(npz_path, annotations_path, test_vids, split='test')
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=1)

        # model initialization
        model = TaskVerificationTransformer(embed_dim=256).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        # training loop
        for epoch in range(30):
            model.train()
            for batch in train_loader:
                feats, labels, masks = batch['features'].to(device), batch['label'].to(device).float(), batch['mask'].to(device)
                optimizer.zero_grad()
                logits = model(feats, masks)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            batch = next(iter(test_loader))
            logits = model(batch['features'].to(device), batch['mask'].to(device))
            pred = (torch.sigmoid(logits) > 0.5).int().item()
            gt = batch['label'].item()
            is_correct = (pred == gt)
            
        results.append(is_correct)
        wandb.log({"fold_accuracy": int(is_correct), "total_running_acc": np.mean(results)})
        run.finish()

        if fold % 10 == 0:
            print(f"Fold {fold}/{len(video_ids)} - Running Accuracy: {np.mean(results):.4f}")

    print(f"Final LOO Accuracy: {np.mean(results):.4f}")

if __name__ == "__main__":
    train_loo('step_embeddings_dataset.npz', 'complete_step_annotations.json')