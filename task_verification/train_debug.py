# debug_gnn.py — esegui con: python -m task_verification.debug_gnn
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from task_verification.dataset_GNN import TaskVerificationGraphDataset, graph_collate_fn
from task_verification.GNN import TaskVerificationGNN

def run_debug():
    # === CONFIG ===
    args = {
        'visual_npz':      'step_embeddings_dataset.npz',
        'text_npz':        'text_task_graphs_v2.npz',
        'graph_zip':       'annotations/task_graphs',
        'annotations_json':'annotations/annotation_json/complete_step_annotations.json',
        'batch_size':      8,
        'epochs':          3,
        'lr':              2e-4,
        'weight_decay':    1e-2,
        'dropout':         0.4,
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # === LOAD DATA ===
    print("\nLoading NPZ files...")
    global_visual = {k.replace('.npy',''): v.astype(np.float32)
                     for k, v in np.load(args['visual_npz']).items()}
    global_text   = {k.replace('.npy',''): v.astype(np.float32)
                     for k, v in np.load(args['text_npz']).items()}

    all_ids   = sorted(global_visual.keys())
    test_id   = all_ids[0]
    train_ids = [v for v in all_ids if v != test_id]

    print(f"\nTest video: {test_id}")
    print(f"Train videos: {len(train_ids)}")

    # === SANITY: data shapes ===
    print("\n=== DATA SANITY ===")
    print(f"visual[{test_id}]: {global_visual[test_id].shape}")   # expect [K, 768]
    print(f"text[{test_id}]:   {global_text[test_id].shape}")     # expect [N, 256]
    prefix = test_id.split('_')[0]
    same   = [v for v in global_text if v.startswith(prefix+'_')]
    identical = all(np.array_equal(global_text[test_id], global_text[v]) for v in same)
    print(f"text features identical within recipe {prefix}: {identical}")  # must be True

    # === DATASET ===
    train_ds = TaskVerificationGraphDataset(
        global_visual, global_text,
        args['graph_zip'], args['annotations_json'], train_ids, split='train'
    )
    print(f"\nTrain dataset size: {len(train_ds)}")
    print(f"Label distribution: {sum(1 for i in range(len(train_ds)) if train_ds[i]['label'].item()==1)} positive, "
          f"{sum(1 for i in range(len(train_ds)) if train_ds[i]['label'].item()==0)} negative")

    # === INSPECT FIRST SAMPLE ===
    print("\n=== FIRST SAMPLE INSPECTION ===")
    sample = train_ds[0]
    print(f"video_id:       {sample['video_id']}")
    print(f"visual_features:{sample['visual_features'].shape}")   # [K, 768]
    print(f"text_features:  {sample['text_features'].shape}")     # [N, 256]
    print(f"edge_index:     {sample['edge_index'].shape}")         # [2, E]
    print(f"label:          {sample['label'].item()}")

    ei = sample['edge_index']
    N  = sample['text_features'].shape[0]
    if ei.numel() > 0:
        max_idx = ei.max().item()
        print(f"edge max_idx={max_idx}, N={N} → {'OK' if max_idx < N else '*** OUT OF BOUNDS ***'}")
    else:
        print("edge_index: EMPTY (no edges)")

    # === DATALOADER BATCH INSPECTION ===
    train_loader = DataLoader(train_ds, batch_size=args['batch_size'], shuffle=True,
                              collate_fn=graph_collate_fn, num_workers=0)
    batch = next(iter(train_loader))

    print("\n=== BATCH INSPECTION ===")
    vis  = batch['visual_features']   # [B, K_max, 768]
    txt  = batch['text_features']     # [B, N_max, 256]
    vm   = batch['visual_mask']       # [B, K_max]
    tm   = batch['text_mask']         # [B, N_max]
    ei_l = batch['edge_indices']
    lbl  = batch['labels']

    print(f"visual: {vis.shape}, nonzero rows: {(vis.abs().sum(-1)>0.01).float().mean():.3f}")
    print(f"text:   {txt.shape}, nonzero rows: {(txt.abs().sum(-1)>0.01).float().mean():.3f}")
    print(f"labels: {lbl.numpy()}")

    for i, e in enumerate(ei_l):
        n_nodes = int(tm[i].sum().item())
        if e.numel() > 0:
            ok = e.max().item() < n_nodes
            print(f"  edge[{i}]: shape={e.shape}, max_idx={e.max().item()}, n_nodes={n_nodes} → {'OK' if ok else '*** OUT OF BOUNDS ***'}")
        else:
            print(f"  edge[{i}]: EMPTY, n_nodes={n_nodes}")

    # === MODEL INIT ===
    model = TaskVerificationGNN(dropout=args['dropout']).to(device)
    vis_d  = vis.to(device)
    txt_d  = txt.to(device)
    vm_d   = vm.to(device)
    tm_d   = tm.to(device)
    lbl_d  = lbl.to(device)

    # === FORWARD PASS BEFORE TRAINING ===
    print("\n=== FORWARD PASS (epoch 0, random init) ===")
    model.eval()
    with torch.no_grad():
        realized, align_loss = model.node_realizer(vis_d, txt_d, vm_d, tm_d)
        logits, _            = model(vis_d, txt_d, vm_d, tm_d, ei_l)
        probs                = torch.sigmoid(logits)

    print(f"realized: mean={realized.mean():.4f} std={realized.std():.6f}")
    print(f"logits:   {logits.detach().cpu().numpy().round(3)}")
    print(f"logits std: {logits.std():.6f}  ← deve essere > 0.1 dopo training")
    print(f"probs:    {probs.detach().cpu().numpy().round(3)}")
    print(f"align_loss: {align_loss.item():.4f}")

    # === TRAINING LOOP (3 epochs, 1 batch) ===
    print("\n=== TRAINING (3 epochs) ===")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    criterion = nn.BCEWithLogitsLoss()
    ls = 0.1

    model.train()
    for epoch in range(1, args['epochs']+1):
        smoothed = lbl_d * (1-ls) + (1-lbl_d) * ls
        optimizer.zero_grad()
        logits, al = model(vis_d, txt_d, vm_d, tm_d, ei_l)
        loss = criterion(logits, smoothed) + 0.01 * al
        loss.backward()
        
        # Gradient norms
        grad_norms = {name: param.grad.abs().max().item()
                      for name, param in model.named_parameters()
                      if param.grad is not None}
        
        optimizer.step()
        probs = torch.sigmoid(logits)
        print(f"  Epoch {epoch}: loss={loss.item():.4f} probs={probs.detach().cpu().numpy().round(3)} labels={lbl_d.cpu().numpy()}")

    # === GRADIENT ANALYSIS ===
    print("\n=== GRADIENT ANALYSIS (last epoch) ===")
    sorted_grads = sorted(grad_norms.items(), key=lambda x: -x[1])
    for name, gmax in sorted_grads:
        status = "OK" if gmax > 1e-6 else "*** DEAD ***"
        print(f"  {name}: {gmax:.2e} {status}")

    # === FORWARD PASS AFTER TRAINING ===
    print("\n=== FORWARD PASS (after 3 epochs) ===")
    model.eval()
    with torch.no_grad():
        logits2, _ = model(vis_d, txt_d, vm_d, tm_d, ei_l)
        probs2 = torch.sigmoid(logits2)
    print(f"logits: {logits2.detach().cpu().numpy().round(3)}")
    print(f"probs:  {probs2.detach().cpu().numpy().round(3)}")
    print(f"labels: {lbl_d.cpu().numpy()}")
    
    sep = probs2[lbl_d==1].mean().item() - probs2[lbl_d==0].mean().item()
    print(f"\nSeparazione prob GT=1 vs GT=0: {sep:.4f}  ← deve essere > 0.05 per imparare")
    if abs(sep) < 0.02:
        print("*** COLLAPSE CONFERMATO: il modello non separa le classi ***")
    else:
        print("OK: il modello sta imparando a separare le classi")

if __name__ == '__main__':
    run_debug()