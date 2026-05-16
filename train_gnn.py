import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import warnings
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from core.models.graph_model import GraphClassifier
from torch.utils.data import DataLoader, Subset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_graph_batch(batch):
    """
    Collate function to align graphs of different sizes into 3D dense tensor using padding.
    Keeps the virtual node dynamically positioned at the absolute end (-1) of the sequence dimension.
    """
    batch_size = len(batch)
    
    # Identify the maximum dimensions within this specific batch
    max_visual = max(sample['v'].size(0) for sample in batch)
    
    # sample['t'] contains [Num_Real_Nodes + 1, Feature_Dim] due to the virtual node
    max_real_nodes = max(sample['t'].size(0) - 1 for sample in batch)
    max_total_nodes = max_real_nodes + 1 
    
    v_dim = batch[0]['v'].size(-1)
    t_dim = batch[0]['t'].size(-1)
    
    # Initialize 3D padded tensors
    batched_v = torch.zeros(batch_size, max_visual, v_dim)
    batched_t = torch.zeros(batch_size, max_total_nodes, t_dim)
    batched_adj = torch.zeros(batch_size, max_total_nodes, max_total_nodes)
    batched_y = torch.zeros(batch_size)
    
    # Binary mask: True for valid nodes (real + virtual), False for dummy padding nodes
    batched_mask = torch.zeros(batch_size, max_total_nodes, dtype=torch.bool)
    ids = []
    
    for i, sample in enumerate(batch):
        ids.append(sample['id'])
        num_v = sample['v'].size(0)
        num_real = sample['t'].size(0) - 1
        
        # Populate visual features
        batched_v[i, :num_v, :] = sample['v']
        
        # Populate text features (Real nodes at the beginning, Virtual Node at the absolute end)
        batched_t[i, :num_real, :] = sample['t'][:-1, :]
        batched_t[i, -1, :] = sample['t'][-1, :] # Virtual Node mapped to index -1
        
        # Reconstruct Adjacency Matrix topology for padded dimensions
        # Real to Real connections
        batched_adj[i, :num_real, :num_real] = sample['adj'][:num_real, :num_real]
        # Real to Virtual connections (last column)
        batched_adj[i, :num_real, -1] = sample['adj'][:num_real, -1]
        # Virtual to Real connections (last row)
        batched_adj[i, -1, :num_real] = sample['adj'][-1, :num_real]
        # Virtual self-loop
        batched_adj[i, -1, -1] = sample['adj'][-1, -1]
        
        # Target Label
        batched_y[i] = sample['y']
        
        # Mask Definition
        batched_mask[i, :num_real] = True
        batched_mask[i, -1] = True # Virtual node is always a valid node
        
    return {
        'ids': ids,
        'v': batched_v,
        't': batched_t,
        'adj': batched_adj,
        'mask': batched_mask,
        'y': batched_y
    }

def evaluate_metrics(y_true, y_pred, y_scores=None, average='macro') -> dict:
    """
    Computes multiple classification metrics to evaluate the global performance.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    
    # Compute AUROC only if probabilities are provided
    if y_scores is not None:
        try:
            metrics["auroc"] = roc_auc_score(y_true, y_scores, multi_class='ovr', average=average)
        except ValueError:
            metrics["auroc"] = float('nan')
            
    return metrics

def load_graph_data(visual_npz, text_npz, annotations_file):
    print("Loading visual and textual features...")
    v_data = np.load(visual_npz)
    t_data = np.load(text_npz)
    
    with open(annotations_file, 'r') as f:
        error_annotations = json.load(f)
        
    # Exact label mapping as in Substep 2
    recipe_labels = {}
    for recording in error_annotations:
        rec_id = recording['recording_id']
        has_error = any("errors" in step for step in recording.get('step_annotations', []))
        recipe_labels[rec_id] = 1 if has_error else 0
        
    dataset = []
    groups = []
    labels = []
    
    for rec_id in v_data.files:
        if rec_id in t_data:
            v_feats = torch.tensor(v_data[rec_id], dtype=torch.float32).to(device)
            t_feats = torch.tensor(t_data[rec_id], dtype=torch.float32).to(device)
            
            num_nodes = t_feats.shape[0]
            
            # Create a Virtual Node feature (initialized as the mean of text features)
            virtual_feat = t_feats.mean(dim=0, keepdim=True)
            t_feats_extended = torch.cat([t_feats, virtual_feat], dim=0)
            
            num_nodes_total = num_nodes + 1
            adj = torch.eye(num_nodes_total).to(device)
            
            # Sequential bidirectional edges for real steps
            for i in range(num_nodes - 1):
                adj[i, i+1] = 1.0 
                adj[i+1, i] = 1.0 
                
            # Connect the Virtual Node (last index) to all real nodes
            for i in range(num_nodes):
                adj[i, -1] = 1.0 # Edge from real node to virtual node
                adj[-1, i] = 1.0 # Edge from virtual node to real node
            
            y = recipe_labels.get(rec_id, 0)
            
            dataset.append({
                'id': rec_id,
                'v': v_feats,
                't': t_feats_extended, # Use extended features
                'adj': adj,
                'y': torch.tensor(y, dtype=torch.float32).to(device)
            })
            groups.append(rec_id)
            labels.append(y)
            
    return dataset, np.array(groups), np.array(labels)

def train_gnn_logo(dataset, groups, labels, num_epochs=15, batch_size=16, lr=1e-4, hidden_dim=256, num_layers=2, dropout=0.4, weight_decay=1e-5):
    logo = LeaveOneGroupOut()
    all_preds = []
    all_targets = []
    all_scores = []
    
    indices = np.arange(len(dataset))
    
    # Initialize dimensions based on the first sample
    v_dim = dataset[0]['v'].shape[-1]
    t_dim = dataset[0]['t'].shape[-1]
    
    for fold, (train_idx, test_idx) in enumerate(logo.split(indices, labels, groups=groups)):
        test_recipe_id = groups[test_idx[0]]
        print(f"--- Fold {fold + 1}/{len(groups)} (Test Recipe: {test_recipe_id}) ---")
        
        # Count positive and negative samples in the current training set
        train_labels = [labels[i] for i in train_idx]
        num_pos = sum(train_labels)
        num_neg = len(train_labels) - num_pos

        # Calculate pos_weight (Negative count / Positive count)
        # This heavily penalizes the model if it gets the minority class wrong
        pos_weight_val = torch.tensor([num_neg / (num_pos + 1e-5)], dtype=torch.float32).to(device)
        
        # Create a batched training sub-dataset for the current cross-validation fold
        train_subdataset = Subset(dataset, train_idx)
        train_loader = DataLoader(
            train_subdataset, 
            batch_size=batch_size,
            shuffle=True, 
            collate_fn=collate_graph_batch,
            pin_memory=True,
            num_workers=4
        )
        
        # Initialize a fresh model for each fold
        model = GraphClassifier(
            visual_dim=v_dim, 
            text_dim=t_dim, 
            hidden_dim=hidden_dim, 
            num_layers=num_layers, 
            dropout_prob=dropout
        ).to(device)
        model = torch.compile(model) 
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_val)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Initialize the Gradient Scaler for Mixed Precision
        scaler = torch.amp.GradScaler('cuda')
        
        # Training
        model.train()
        for _ in range(num_epochs):
            epoch_loss = 0.0
            
            for batch in train_loader:
                # Set to none is faster than writing zeros
                optimizer.zero_grad(set_to_none=True)
                
                # Fetch padded inputs directly from the collated dictionary mapping
                v_inputs = batch['v'].to(device, non_blocking=True)
                t_inputs = batch['t'].to(device, non_blocking=True)
                adj_inputs = batch['adj'].to(device, non_blocking=True)
                targets = batch['y'].to(device, non_blocking=True)
                
                # Automatic Mixed Precision context manager
                with torch.amp.autocast('cuda'):
                    # Forward pass processes graphs simultaneously
                    logits = model(v_inputs, t_inputs, adj_inputs)
                    loss = criterion(logits, targets)
                
                # Backward pass via scaler
                scaler.scale(loss).backward()
                
                # Total gradient clipping protection (must unscale first)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step and scaler update
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss += loss.item() * len(batch['ids'])
            
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_sample = dataset[test_idx[0]]
            
            # Add a batch dimension of 1 for the refactored 3D model
            v_test = test_sample['v'].unsqueeze(0).to(device)
            t_test = test_sample['t'].unsqueeze(0).to(device)
            adj_test = test_sample['adj'].unsqueeze(0).to(device)
            
            # Forward pass with batched inputs
            test_logits = model(v_test, t_test, adj_test)
            
            prob = torch.sigmoid(test_logits).item()
            
            pred = 1.0 if prob >= 0.5 else 0.0
            target = test_sample['y'].item()
            
            all_preds.append(pred)
            all_targets.append(target)
            all_scores.append(prob)
            
            print(f"    Target: {int(target)} | Prediction: {int(pred)} | Correct: {pred == target}")
            
    # Global evaluation
    total_metrics = evaluate_metrics(np.array(all_targets), np.array(all_preds), y_scores=np.array(all_scores))
    
    print(f"""
Global GNN (Substep 4) LOGO Results ->
    Acc:   {total_metrics['accuracy']:.4f}
    Prec:  {total_metrics['precision']:.4f}
    Rec:   {total_metrics['recall']:.4f}
    F1:    {total_metrics['f1_score']:.4f}
    AUROC: {total_metrics['auroc']:.4f}""")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-step Action Graph Neural Network")
    parser.add_argument("--visual_npz", type=str, required=True, help="Path to visual step embeddings")
    parser.add_argument("--text_npz", type=str, required=True, help="Path to text task graph embeddings")
    parser.add_argument("--annotations", type=str, default="annotations/annotation_json/error_annotations.json")
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=15, help="Epochs per fold")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for the DataLoader")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate") # Fixed type to float
    
    # Architecture and Regularization hyperparameters
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension size")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of GNN message passing layers")
    parser.add_argument("--dropout", type=float, default=0.4, help="Dropout probability")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="L2 regularization penalty for Adam")
    args = parser.parse_args()
    
    print("\n[Starting Graph Neural Network Training]")
    print(f"Hardware utilized: {device.type.upper()}")
    
    dataset, groups, labels = load_graph_data(args.visual_npz, args.text_npz, args.annotations)
    print(f"Dataset loaded: {len(dataset)} graphs.")
    
    train_gnn_logo(
        dataset, groups, labels, 
        num_epochs=args.epochs, 
        batch_size=args.batch_size, 
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        weight_decay=args.weight_decay
    )