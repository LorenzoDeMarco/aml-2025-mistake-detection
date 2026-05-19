# debug_gnn.py
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from task_verification.dataset_GNN import TaskVerificationGraphDataset, graph_collate_fn
from task_verification.GNN import TaskVerificationGNN

def run_debug():
    args = {
        'visual_npz':      'step_embeddings_dataset.npz',
        'text_npz':        'text_task_graphs_v2.npz',
        'graph_zip':       'task_graphs',
        'annotations_json':'annotations/annotation_json/complete_step_annotations.json',
        'batch_size':      8,
        'epochs':          20,
        'lr':              1e-4,
        'weight_decay':    1e-2,
        'dropout':         0.5,
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("\nLoading NPZ files...")
    global_visual = {k.replace('.npy',''): v.astype(np.float32)
                     for k, v in np.load(args['visual_npz']).items()}
    global_text   = {k.replace('.npy',''): v.astype(np.float32)
                     for k, v in np.load(args['text_npz']).items()}

    all_ids   = sorted(global_visual.keys())
    test_id   = all_ids[0]
    train_ids = [v for v in all_ids if v != test_id]

    print(f"Test video: {test_id}")

    train_ds = TaskVerificationGraphDataset(
        global_visual, global_text,
        args['graph_zip'], args['annotations_json'], train_ids, split='train'
    )
    train_loader = DataLoader(train_ds, batch_size=args['batch_size'], shuffle=True,
                              collate_fn=graph_collate_fn, num_workers=0)

    n_pos = sum(1 for i in range(len(train_ds)) if train_ds[i]['label'].item() == 1)
    n_neg = len(train_ds) - n_pos
    print(f"Train: {len(train_ds)} samples — {n_pos} pos, {n_neg} neg")

    model = TaskVerificationGNN(dropout=args['dropout']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['epochs'], eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss()
    ls = 0.1

    print("\n=== TRAINING ===")
    for epoch in range(1, args['epochs'] + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            vis   = batch["visual_features"].to(device)
            txt   = batch["text_features"].to(device)
            vm    = batch["visual_mask"].to(device)
            tm    = batch["text_mask"].to(device)
            ei    = batch["edge_indices"]
            lbl   = batch["labels"].to(device)

            smoothed = lbl * (1 - ls) + (1 - lbl) * ls
            optimizer.zero_grad()
            logits, al = model(vis, txt, vm, tm, ei)
            loss = criterion(logits, smoothed) + 0.01 * al
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * vis.size(0)

        scheduler.step()
        avg_loss = total_loss / len(train_ds)

        # Check sul training set ogni 5 epoche
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            all_probs, all_gts = [], []
            with torch.no_grad():
                for batch in train_loader:
                    vis = batch["visual_features"].to(device)
                    txt = batch["text_features"].to(device)
                    vm  = batch["visual_mask"].to(device)
                    tm  = batch["text_mask"].to(device)
                    ei  = batch["edge_indices"]
                    lbl = batch["labels"]

                    logits, _ = model(vis, txt, vm, tm, ei)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    all_probs.extend(probs)
                    all_gts.extend(lbl.numpy())

            all_probs = np.array(all_probs)
            all_gts   = np.array(all_gts)
            sep  = all_probs[all_gts==1].mean() - all_probs[all_gts==0].mean()
            acc  = accuracy_score(all_gts, (all_probs >= 0.5).astype(int))
            print(f"  Epoch {epoch:>2} | loss={avg_loss:.4f} | "
                  f"train_acc={acc:.3f} | "
                  f"mean_GT0={all_probs[all_gts==0].mean():.4f} | "
                  f"mean_GT1={all_probs[all_gts==1].mean():.4f} | "
                  f"sep={sep:+.4f}")
            model.train()

    print("\n=== FINAL TRAIN SET CHECK ===")
    model.eval()
    all_probs, all_gts = [], []
    with torch.no_grad():
        for batch in train_loader:
            vis = batch["visual_features"].to(device)
            txt = batch["text_features"].to(device)
            vm  = batch["visual_mask"].to(device)
            tm  = batch["text_mask"].to(device)
            ei  = batch["edge_indices"]
            lbl = batch["labels"]
            logits, _ = model(vis, txt, vm, tm, ei)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_gts.extend(lbl.numpy())

    all_probs = np.array(all_probs)
    all_gts   = np.array(all_gts)
    sep = all_probs[all_gts==1].mean() - all_probs[all_gts==0].mean()
    acc = accuracy_score(all_gts, (all_probs >= 0.5).astype(int))
    print(f"  train_acc={acc:.4f}")
    print(f"  mean prob GT=0: {all_probs[all_gts==0].mean():.4f} ± {all_probs[all_gts==0].std():.4f}")
    print(f"  mean prob GT=1: {all_probs[all_gts==1].mean():.4f} ± {all_probs[all_gts==1].std():.4f}")
    print(f"  separazione:    {sep:+.4f}")

    if sep > 0.05:
        print("  ✓ Il modello IMPARA sul training set — problema è overfitting/generalizzazione")
    elif sep > 0.01:
        print("  △ Segnale debole — il modello impara pochissimo anche in training")
    else:
        print("  ✗ COLLAPSE anche in training — problema architetturale o di loss")

    # Prob distribution
    print("\n  Prob distribution training set:")
    bins = [0, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 1.0]
    for i in range(len(bins)-1):
        mask = (all_probs >= bins[i]) & (all_probs < bins[i+1])
        n = mask.sum()
        if n > 0:
            gt1 = all_gts[mask].sum()
            print(f"    [{bins[i]:.2f},{bins[i+1]:.2f}): {n:3d} samples, GT=1: {gt1:3d} ({gt1/n*100:.0f}%)")

if __name__ == '__main__':
    run_debug()