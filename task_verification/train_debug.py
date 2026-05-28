# train_debug.py
# Confronto diretto: BCE+Cosine vs FocalLoss+Cosine
# Su un mini-LOGO reale (3 fold) per una risposta rapida e affidabile.
# Tempo atteso: ~6 minuti totali.

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score

try:
    from task_verification.dataset_GNN import TaskVerificationGraphDataset, graph_collate_fn
    from task_verification.GNN import TaskVerificationGNN
except ModuleNotFoundError:
    from dataset_GNN import TaskVerificationGraphDataset, graph_collate_fn
    from GNN import TaskVerificationGNN


# ─── Loss functions ──────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt  = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


# ─── Single fold training ────────────────────────────────────────────────────

def train_one_fold(train_ids, test_ids, global_visual, global_text, args, use_focal):
    """
    Trains and evaluates a single fold.
    Returns dict with AUROC, F1, Recall, Accuracy, prob separation.
    """
    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])
    torch.cuda.manual_seed_all(args['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds = TaskVerificationGraphDataset(
        preloaded_visual=global_visual, preloaded_text=global_text,
        graph_zip_path=args['graph_zip'], annotations_path=args['annotations_json'],
        video_ids=train_ids, split='train'
    )
    test_ds = TaskVerificationGraphDataset(
        preloaded_visual=global_visual, preloaded_text=global_text,
        graph_zip_path=args['graph_zip'], annotations_path=args['annotations_json'],
        video_ids=test_ids, split='test'
    )

    train_loader = DataLoader(train_ds, batch_size=args['batch_size'], shuffle=True,
                              num_workers=2, collate_fn=graph_collate_fn, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args['batch_size'], shuffle=False,
                              num_workers=2, collate_fn=graph_collate_fn, pin_memory=True)

    model = TaskVerificationGNN(dropout=args['dropout']).to(device)

    proj_params, base_params = [], []
    for name, param in model.named_parameters():
        if any(k in name for k in ['sim_visual_proj', 'sim_text_proj', 'logit_scale']):
            proj_params.append(param)
        else:
            base_params.append(param)

    optimizer = optim.AdamW([
        {'params': base_params,  'lr': args['lr']},
        {'params': proj_params,  'lr': args['lr'] * 5.0}
    ], weight_decay=args['weight_decay'])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['epochs'], eta_min=1e-6)

    if use_focal:
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        label_fn  = lambda lbl: lbl.float()           # no smoothing with Focal
    else:
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        ls = 0.1
        label_fn  = lambda lbl: lbl * (1 - ls) + (1 - lbl) * ls   # label smoothing with BCE

    for epoch in range(1, args['epochs'] + 1):
        align_weight = max(0.1, 1.0 * (0.8 ** (epoch - 1)))
        model.train()
        for batch in train_loader:
            vis  = batch["visual_features"].to(device)
            txt  = batch["text_features"].to(device)
            vm   = batch["visual_mask"].to(device)
            tm   = batch["text_mask"].to(device)
            ei   = batch["edge_indices"]
            nd   = batch["node_depths"].to(device)
            lbl  = batch["labels"].to(device)

            optimizer.zero_grad()
            logits, align_loss = model(vis, txt, vm, tm, ei, nd)
            loss = criterion(logits, label_fn(lbl)) + align_weight * align_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

    # ── Evaluation ──
    model.eval()
    all_probs, all_gts = [], []
    with torch.no_grad():
        for batch in test_loader:
            vis = batch["visual_features"].to(device)
            txt = batch["text_features"].to(device)
            vm  = batch["visual_mask"].to(device)
            tm  = batch["text_mask"].to(device)
            ei  = batch["edge_indices"]
            nd  = batch["node_depths"].to(device)
            lbl = batch["labels"]

            logits, _ = model(vis, txt, vm, tm, ei, nd)
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_gts.extend(lbl.numpy())

    probs = np.array(all_probs)
    gts   = np.array(all_gts)
    preds = (probs >= 0.5).astype(int)

    sep = probs[gts == 1].mean() - probs[gts == 0].mean() if gts.sum() > 0 and (gts == 0).sum() > 0 else 0.0

    try:
        auroc = roc_auc_score(gts, probs)
    except Exception:
        auroc = float('nan')

    return {
        'auroc': auroc,
        'f1':    f1_score(gts, preds, zero_division=0),
        'rec':   recall_score(gts, preds, zero_division=0),
        'acc':   accuracy_score(gts, preds),
        'sep':   sep,
        'n_test': len(gts)
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def run_debug():
    args = {
        'visual_npz':       'step_embeddings.npz',
        'text_npz':         'text_task_graphs_v2.npz',
        'graph_zip':        'annotations/task_graphs',
        'annotations_json': 'annotations/annotation_json/complete_step_annotations.json',
        'batch_size': 16,
        'epochs':     25,       # 25 epoche: abbastanza per vedere convergenza, veloce (~1 min/fold)
        'lr':         2e-4,
        'weight_decay': 1e-2,
        'dropout':    0.4,
        'seed':       42,
    }

    print("Loading NPZ files into RAM...")
    global_visual = {k.replace('.npy', ''): v.astype(np.float32)
                     for k, v in np.load(args['visual_npz']).items()}
    global_text   = {k.replace('.npy', ''): v.astype(np.float32)
                     for k, v in np.load(args['text_npz']).items()}

    all_ids = sorted(list(global_visual.keys()))

    # ── Mini-LOGO: 3 fold rappresentativi ──────────────────────────────────
    # Scegliamo recipe con caratteristiche diverse:
    #   recipe 1  → alta separazione nelle run precedenti (buona baseline)
    #   recipe 17 → separazione negativa nelle run precedenti (caso difficile)
    #   recipe 20 → separazione medio-alta (caso medio)
    TEST_RECIPES = ['1', '17', '20']

    recipe_groups = {}
    for vid in all_ids:
        r = vid.split('_')[0]
        recipe_groups.setdefault(r, []).append(vid)

    print(f"\n{'='*80}")
    print(f"  Mini-LOGO Debug: 3 fold x 2 loss = 6 training run (~6 min totali)")
    print(f"  Fold: recipe {TEST_RECIPES}  |  Epoche: {args['epochs']}  |  Seed: {args['seed']}")
    print(f"{'='*80}\n")
    print(f"{'Recipe':<10} | {'Loss':<10} | {'AUROC':<7} | {'F1':<7} | {'Rec':<7} | {'Acc':<7} | {'Sep':>7} | {'N_test'}")
    print("-" * 80)

    results = {'bce': {}, 'focal': {}}

    for recipe_id in TEST_RECIPES:
        if recipe_id not in recipe_groups:
            print(f"  Recipe {recipe_id} not found, skipping.")
            continue

        test_ids  = recipe_groups[recipe_id]
        train_ids = [v for v in all_ids if v not in test_ids]

        for loss_name, use_focal in [('BCE+LS', False), ('Focal', True)]:
            r = train_one_fold(train_ids, test_ids, global_visual, global_text, args, use_focal)
            results['bce' if loss_name == 'BCE+LS' else 'focal'][recipe_id] = r
            print(f"  {recipe_id:<8} | {loss_name:<10} | {r['auroc']:.4f} | {r['f1']:.4f} | {r['rec']:.4f} | {r['acc']:.4f} | {r['sep']:+.4f} | {r['n_test']}")

    # ── Sommario aggregato ─────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  SOMMARIO (media sui 3 fold)")
    print(f"{'='*80}")

    for loss_name, key in [('BCE + LabelSmoothing', 'bce'), ('Focal Loss (no smooth)', 'focal')]:
        fold_res = results[key]
        if not fold_res:
            continue
        metrics = {m: np.mean([fold_res[r][m] for r in fold_res]) for m in ['auroc', 'f1', 'rec', 'acc', 'sep']}
        print(f"  {loss_name:<30} | AUROC={metrics['auroc']:.4f} | F1={metrics['f1']:.4f} | "
              f"Rec={metrics['rec']:.4f} | Acc={metrics['acc']:.4f} | Sep={metrics['sep']:+.4f}")

    print(f"\n{'='*80}")
    print("  INTERPRETAZIONE:")
    print("  - Sep > +0.10  →  buona discriminazione")
    print("  - Sep < 0      →  inversione (failure mode)")
    print("  - Recall è la metrica prioritaria per mistake detection")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    run_debug()
