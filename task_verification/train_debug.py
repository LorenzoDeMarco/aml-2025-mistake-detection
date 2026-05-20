# debug_gnn.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from task_verification.dataset_GNN import TaskVerificationGraphDataset, graph_collate_fn
from task_verification.GNN import TaskVerificationGNN

def run_debug():
    args = {
        'visual_npz':      'step_embeddings_dataset.npz',
        'text_npz':        'text_task_graphs.npz',  # Assicurati che sia quello corretto
        'graph_zip':       'task_graphs',
        'annotations_json':'complete_step_annotations.json',
        'batch_size':      16, # Impostato a 16 per bilanciare l'Ungherese su CPU
        'epochs':          25, # Qualche epoca in più per vedere bene la divergenza
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

    all_ids   = sorted(global_visual.keys())
    test_id   = all_ids[0]
    
    # Nel debug script, overfittiamo volutamente sul training set
    # per verificare che l'architettura sia in grado di imparare (capacità rappresentativa)
    train_ids = [vid for vid in all_ids if vid != test_id]

    # Prendi un sottoinsieme per fare debugging veloce (es. 64 video)
    # Rimuovi lo slicing [:64] se vuoi fare il debug su tutto il dataset
    debug_train_ids = train_ids[:64]
    
    print(f"Creating Dataset with {len(debug_train_ids)} videos...")
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
    optimizer = optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    label_smoothing = 0.1

    print("\nStarting Debug Training Loop (Overfitting Test)...")
    print("-" * 75)
    print(f"{'Epoch':<6} | {'Tot Loss':<10} | {'Cls Loss':<10} | {'InfoNCE':<10} | {'Separation (GT1 - GT0)'}")
    print("-" * 75)

    for epoch in range(1, args['epochs'] + 1):
        model.train()
        train_loss = 0.0
        total_cls_loss = 0.0
        total_align_loss = 0.0
        
        # Variabili per tracciare la separazione delle probabilità live
        epoch_probs = []
        epoch_gts = []

        for batch in train_loader:
            vis = batch["visual_features"].to(device)
            txt = batch["text_features"].to(device)
            vm  = batch["visual_mask"].to(device)
            tm  = batch["text_mask"].to(device)
            ei  = batch["edge_indices"]
            lbl = batch["labels"].to(device)

            # Correzione matematica del Label Smoothing: target -> [0.1, 0.9]
            smoothed_labels = lbl * (1.0 - label_smoothing) + label_smoothing / 2.0

            optimizer.zero_grad()
            
            # Cattura sia i logit che la Contrastive InfoNCE Loss
            logits, align_loss = model(vis, txt, vm, tm, ei) 
            
            classification_loss = criterion(logits, smoothed_labels)
            
            # Loss composita
            loss = classification_loss + 0.1 * align_loss 
            
            loss.backward()
            
            # Gradient clipping di sicurezza
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Tracciamento loss
            bsz = vis.size(0)
            train_loss += loss.item() * bsz
            total_cls_loss += classification_loss.item() * bsz
            total_align_loss += align_loss.item() * bsz
            
            # Registriamo le probabilità per l'analisi a fine epoca
            with torch.no_grad():
                probs = torch.sigmoid(logits).cpu().numpy()
                epoch_probs.extend(probs)
                epoch_gts.extend(lbl.cpu().numpy())

        # Calcolo medie dell'epoca
        avg_loss = train_loss / len(train_dataset)
        avg_cls = total_cls_loss / len(train_dataset)
        avg_align = total_align_loss / len(train_dataset)
        
        # Calcolo separazione live
        epoch_probs = np.array(epoch_probs)
        epoch_gts = np.array(epoch_gts)
        
        mean_prob_gt1 = epoch_probs[epoch_gts == 1].mean() if len(epoch_probs[epoch_gts == 1]) > 0 else 0
        mean_prob_gt0 = epoch_probs[epoch_gts == 0].mean() if len(epoch_probs[epoch_gts == 0]) > 0 else 0
        live_sep = mean_prob_gt1 - mean_prob_gt0

        # Log avanzato visivo
        print(f"{epoch:<6} | {avg_loss:<10.4f} | {avg_cls:<10.4f} | {avg_align:<10.4f} | {live_sep:+.4f}")

    # ==========================================
    # VALUTAZIONE FINALE SUL TRAINING (OVERFIT)
    # ==========================================
    print("-" * 75)
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
            lbl = batch["labels"]
            
            logits, _ = model(vis, txt, vm, tm, ei)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_gts.extend(lbl.numpy())

    all_probs = np.array(all_probs)
    all_gts   = np.array(all_gts)
    sep = all_probs[all_gts==1].mean() - all_probs[all_gts==0].mean()
    acc = accuracy_score(all_gts, (all_probs >= 0.5).astype(int))
    
    print("\n🔍 RISULTATI ANALISI OVERFITTING:")
    print(f"  Training Accuracy: {acc:.4f}")
    print(f"  Media Probabilità (Video Corretti, GT=0): {all_probs[all_gts==0].mean():.4f} ± {all_probs[all_gts==0].std():.4f}")
    print(f"  Media Probabilità (Video con Errore, GT=1): {all_probs[all_gts==1].mean():.4f} ± {all_probs[all_gts==1].std():.4f}")
    print(f"  Distanza di Separazione: {sep:+.4f}")

    if sep > 0.05:
        print("\n  ✅ SUCCESSO: La Contrastive Loss funziona! Il modello sta distanziando le probabilità. L'architettura è salva.")
    elif sep > 0.01:
        print("\n  ⚠️ ALLERTA: Segnale debole. Il modello impara, ma fatica a polarizzare i logit. Possibile tuning necessario su lr o temperatura.")
    else:
        print("\n  ❌ FALLIMENTO: COLLASSO DELLE RAPPRESENTAZIONI. Il modello sta ancora predicendo 0.53 per tutti. Bisogna rivedere il layer di fusione.")

if __name__ == '__main__':
    run_debug()