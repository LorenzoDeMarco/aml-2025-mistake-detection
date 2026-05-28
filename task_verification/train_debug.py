# debug_gnn.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

try:
    from task_verification.dataset_GNN import TaskVerificationGraphDataset, graph_collate_fn
    from task_verification.GNN import TaskVerificationGNN
except ModuleNotFoundError:
    from dataset_GNN import TaskVerificationGraphDataset, graph_collate_fn
    from GNN import TaskVerificationGNN

# --- NUOVA FOCAL LOSS ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def run_debug():
    args = {
        'visual_npz':      'step_embeddings.npz',
        'text_npz':        'text_task_graphs_v2.npz',  
        'graph_zip':       'annotations/task_graphs',
        'annotations_json':'annotations/annotation_json/complete_step_annotations.json',
        'batch_size':      16, 
        'epochs':          25, 
        'lr':              2e-4,
        'weight_decay':    1e-2,
        'dropout':         0.4,
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("\nLoading NPZ files into RAM...")
    global_visual = {k.replace('.npy',''): v.astype(np.float32)
                     for k, v in np.load(args['visual_npz']).items()}
    global_text   = {k.replace('.npy',''): v.astype(np.float32)
                     for k, v in np.load(args['text_npz']).items()}

    all_ids = sorted(list(global_visual.keys()))
    
    # Per il debug veloce, prendiamo solo i primi 64 video
    debug_train_ids = all_ids[:64] 
    
    print(f"Creating Dataset with {len(debug_train_ids)} videos for OVERFITTING TEST...")
    train_dataset = TaskVerificationGraphDataset(
        preloaded_visual=global_visual,
        preloaded_text=global_text,
        graph_zip_path=args['graph_zip'],
        annotations_path=args['annotations_json'],
        video_ids=debug_train_ids,
        split='train'
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args['batch_size'], shuffle=True,
        num_workers=2, collate_fn=graph_collate_fn, pin_memory=True
    )

    model = TaskVerificationGNN(dropout=args['dropout']).to(device)
    
    # --- ACCELERATORE: Differential Learning Rate ---
    projector_params = []
    base_params = []
    fast_learning_keys = ['sim_visual_proj', 'sim_text_proj', 'logit_scale', 'step_positional_encoding']
    
    for name, param in model.named_parameters():
        if any(key in name for key in fast_learning_keys):
            projector_params.append(param)
        else:
            base_params.append(param)
            
    optimizer = optim.AdamW([
        {'params': base_params, 'lr': args['lr']},
        {'params': projector_params, 'lr': args['lr'] * 5.0} 
    ], weight_decay=args['weight_decay'])
    
    # --- SOSTITUZIONE BCE CON FOCAL LOSS E AGGIUNTA SCHEDULER ---
    criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['epochs'], eta_min=1e-6)

    print("\nStarting Debug Training Loop (Focal Loss & Cosine Scheduler Check)...")
    print("-" * 100)
    print(f"{'Epoch':<6} | {'Tot Loss':<10} | {'Cls Loss':<10} | {'InfoNCE':<10} | {'AlgnWt':<6} | {'LR (Base)':<10} | {'Separation'}")
    print("-" * 100)

    for epoch in range(1, args['epochs'] + 1):
        
        current_align_weight = max(0.1, 1.0 * (0.8 ** (epoch - 1)))
        current_lr = optimizer.param_groups[0]['lr']
        
        model.train()
        train_loss = 0.0
        total_cls_loss = 0.0
        total_align_loss = 0.0
        
        epoch_probs = []
        epoch_gts = []

        for batch in train_loader:
            vis = batch["visual_features"].to(device)
            txt = batch["text_features"].to(device)
            vm  = batch["visual_mask"].to(device)
            tm  = batch["text_mask"].to(device)
            ei  = batch["edge_indices"]
            nd  = batch["node_depths"].to(device)
            lbl = batch["labels"].to(device)

            optimizer.zero_grad()
            
            logits, align_loss = model(vis, txt, vm, tm, ei, nd) 
            
            # --- APPLICAZIONE FOCAL LOSS PURA (Niente label smoothing) ---
            classification_loss = criterion(logits, lbl.float())
            
            loss = classification_loss + current_align_weight * align_loss 
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            bsz = vis.size(0)
            train_loss += loss.item() * bsz
            total_cls_loss += classification_loss.item() * bsz
            total_align_loss += align_loss.item() * bsz
            
            with torch.no_grad():
                probs = torch.sigmoid(logits).cpu().numpy()
                epoch_probs.extend(probs)
                epoch_gts.extend(lbl.cpu().numpy())

        # Step dello scheduler alla fine di ogni epoca
        scheduler.step()

        avg_loss = train_loss / len(train_dataset)
        avg_cls = total_cls_loss / len(train_dataset)
        avg_align = total_align_loss / len(train_dataset)
        
        epoch_probs = np.array(epoch_probs)
        epoch_gts = np.array(epoch_gts)
        
        mean_prob_gt1 = epoch_probs[epoch_gts == 1].mean() if len(epoch_probs[epoch_gts == 1]) > 0 else 0
        mean_prob_gt0 = epoch_probs[epoch_gts == 0].mean() if len(epoch_probs[epoch_gts == 0]) > 0 else 0
        live_sep = mean_prob_gt1 - mean_prob_gt0

        print(f"{epoch:<6} | {avg_loss:<10.4f} | {avg_cls:<10.4f} | {avg_align:<10.4f} | {current_align_weight:<6.3f} | {current_lr:<10.6f} | {live_sep:+.4f}")

    # ==========================================
    # VALUTAZIONE FINALE SUL TRAINING (OVERFIT)
    # ==========================================
    print("-" * 100)
    model.eval()
    all_probs = []
    all_gts = []

    with torch.no_grad():
        for batch in train_loader:
            vis = batch["visual_features"].to(device)
            txt = batch["text_features"].to(device)
            vm  = batch["visual_mask"].to(device)
            tm  = batch["text_mask"].to(device)
            ei  = batch["edge_indices"]
            nd  = batch["node_depths"].to(device)
            lbl = batch["labels"]
            
            logits, _ = model(vis, txt, vm, tm, ei, nd)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_gts.extend(lbl.numpy())

    all_probs = np.array(all_probs)
    all_gts   = np.array(all_gts)
    sep = all_probs[all_gts==1].mean() - all_probs[all_gts==0].mean()
    acc = accuracy_score(all_gts, (all_probs >= 0.5).astype(int))
    
    print("\n🔍 RISULTATI ANALISI OVERFITTING (FOCAL LOSS + COSINE LR):")
    print(f"  Training Accuracy: {acc:.4f}")
    print(f"  Media Probabilità (Video Corretti, GT=0): {all_probs[all_gts==0].mean():.4f} ± {all_probs[all_gts==0].std():.4f}")
    print(f"  Media Probabilità (Video con Errore, GT=1): {all_probs[all_gts==1].mean():.4f} ± {all_probs[all_gts==1].std():.4f}")
    print(f"  Distanza di Separazione Finale: {sep:+.4f}")

    if sep > 0.10:
        print("\n  🚀 ECCELLENTE: Il modello overfitta perfettamente i dati! Focal Loss sta discriminando.")
    elif sep > 0.05:
        print("\n  ✅ SUCCESSO: La distanza tra le classi è buona, lo scheduler aiuta la stabilità.")
    elif sep > 0.01:
        print("\n  ⚠️ ALLERTA: Segnale debole. Possibile problema di tuning.")
    else:
        print("\n  ❌ FALLIMENTO: Collasso delle rappresentazioni. Il modello non converge.")

if __name__ == '__main__':
    run_debug()