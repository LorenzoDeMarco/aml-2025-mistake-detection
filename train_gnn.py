import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from core.models.graph_model import GraphClassifier
from torch.utils.data import DataLoader, Subset
import wandb

# Define the hardware accelerator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" class BinaryFocalLoss(nn.Module):
    
    Differentiable Binary Focal Loss optimized for highly skewed sequence recognition.
    Down-weights easy examples to force gradient exploration into hard anomalies.
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, smoothing: float = 0.05, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # x shape / logits shape: (Batch_Size,) or matching targets shape
        
        # Apply mild label smoothing to soften strict targets from {0, 1} to {0.05, 0.95}
        # This prevents the downstream sigmoid function from oversaturating 
        with torch.no_grad():
            smoothed_targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
            
        # Standard stable element-wise binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        probs = torch.sigmoid(logits)
        # Calculate p_t (probability of the ground truth label)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Apply the hard-example mining factor (1 - p_t)^gamma
        focal_modulation = (1 - p_t) ** self.gamma
        loss = focal_modulation * bce_loss
        
        # Apply binary class balancing parameter alpha
        if self.alpha >= 0:
            alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_factor * loss
            
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss """

def collate_graph_batch(batch):
    """
    Collate function to align graphs of varying lengths into dense 3D tensors using padding.
    It dynamically positions the Virtual Node at the absolute end (-1) of the sequence dimension.
    """
    batch_size = len(batch)
    
    # Identify the maximum dimensions within the current batch to define tensor sizes
    max_visual = max(sample['v'].size(0) for sample in batch)
    # The 't' tensor already includes the Virtual Node, so real nodes = total - 1
    max_real_nodes = max(sample['t'].size(0) - 1 for sample in batch)
    max_total_nodes = max_real_nodes + 1 
    
    v_dim = batch[0]['v'].size(-1)
    t_dim = batch[0]['t'].size(-1)
    
    # Initialize empty 3D padded tensors directly on the target device (GPU)
    batched_v = torch.zeros(batch_size, max_visual, v_dim, device=device)
    batched_t = torch.zeros(batch_size, max_total_nodes, t_dim, device=device)
    batched_adj = torch.zeros(batch_size, max_total_nodes, max_total_nodes, device=device)
    batched_y = torch.zeros(batch_size, device=device)
    
    # Binary mask: True for valid nodes (real + virtual), False for padding nodes
    batched_mask = torch.zeros(batch_size, max_total_nodes, dtype=torch.bool, device=device)
    ids = []
    
    for i, sample in enumerate(batch):
        ids.append(sample['id'])
        num_v = sample['v'].size(0)
        num_real = sample['t'].size(0) - 1
        
        # Populate visual features up to the real visual length
        batched_v[i, :num_v, :] = sample['v']
        
        # Populate text features: real nodes at the start, Virtual Node at the very end
        batched_t[i, :num_real, :] = sample['t'][:-1, :]
        batched_t[i, -1, :] = sample['t'][-1, :] 
        
        # Reconstruct the Adjacency Matrix topology for the padded dimensions
        # Real-to-Real edges
        batched_adj[i, :num_real, :num_real] = sample['adj'][:num_real, :num_real]
        # Real-to-Virtual and Virtual-to-Real edges
        batched_adj[i, :num_real, -1] = sample['adj'][:num_real, -1]
        batched_adj[i, -1, :num_real] = sample['adj'][-1, :num_real]
        # Virtual self-loop
        batched_adj[i, -1, -1] = sample['adj'][-1, -1]
        
        # Add self-loops to padding nodes
        # If a row is entirely 0, Softmax computes exp(-inf)/0 = NaN. Self-loops prevent this.
        for pad_idx in range(num_real, max_real_nodes):
            batched_adj[i, pad_idx, pad_idx] = 1.0
        
        # Assign the target label
        batched_y[i] = sample['y']
        
        # Define valid positions in the mask
        batched_mask[i, :num_real] = True
        batched_mask[i, -1] = True 
        
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
    Computes standard classification metrics including the Confusion Matrix.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred) # Raw numpy array for console
    }
    if y_scores is not None:
        try:
            metrics["auroc"] = roc_auc_score(y_true, y_scores, multi_class='ovr', average=average)
        except ValueError:
            metrics["auroc"] = float('nan')
    return metrics

def load_graph_data(visual_npz, text_npz, annotations_file):
    """
    Loads features and constructs the foundational graph topologies.
    For high-VRAM setups, tensors are directly loaded onto the GPU.
    """
    print("Loading visual and textual features...")
    v_data = np.load(visual_npz)
    t_data = np.load(text_npz)
    
    with open(annotations_file, 'r') as f:
        error_annotations = json.load(f)
        
    # Map each recording to a binary label (1 if any step contains an error, else 0)
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
            # Move tensors directly to GPU memory for zero-latency training
            v_feats = torch.tensor(v_data[rec_id], dtype=torch.float32).to(device)
            t_feats = torch.tensor(t_data[rec_id], dtype=torch.float32).to(device)
            
            num_nodes = t_feats.shape[0]
            
            # Create a Virtual Node representation (mean of all step text features)
            virtual_feat = t_feats.mean(dim=0, keepdim=True)
            t_feats_extended = torch.cat([t_feats, virtual_feat], dim=0)
            
            num_nodes_total = num_nodes + 1
            adj = torch.eye(num_nodes_total).to(device)
            
            # Create sequential bidirectional edges for real recipe steps
            for i in range(num_nodes - 1):
                adj[i+1, i] = 1.0
                
            # Fully connect the Virtual Node (last index) to all real nodes
            for i in range(num_nodes):
                adj[-1, i] = 1.0
            
            y = recipe_labels.get(rec_id, 0)
            
            dataset.append({
                'id': rec_id,
                'v': v_feats,
                't': t_feats_extended,
                'adj': adj,
                'y': torch.tensor(y, dtype=torch.float32).to(device)
            })
            groups.append(rec_id)
            labels.append(y)
            
    return dataset, np.array(groups), np.array(labels)

def find_optimal_threshold(y_true, y_scores):
    """
    Evaluates multiple thresholds to find the one that maximizes the F1-score.
    Returns the best threshold and its corresponding F1-score.
    """
    best_threshold = 0.5
    best_f1 = 0.0
    
    # Test thresholds from 0.1 to 0.9 with a 0.05 step
    thresholds = np.arange(0.1, 0.95, 0.01)
    
    for th in thresholds:
        # Convert probabilities to binary predictions based on current threshold
        y_pred = (y_scores >= th).astype(int)
        current_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = th
            
    return best_threshold, best_f1

def train_gnn_logo(
    dataset, 
    groups, 
    labels, 
    num_epochs=15, 
    batch_size=16, 
    lr=1e-4, 
    hidden_dim=256, 
    num_layers=2, 
    dropout=0.4, 
    weight_decay=1e-5, 
    th=0.5
):
    """
    Main training loop implementing Leave-One-Group-Out (LOGO) cross-validation.
    """
    logo = LeaveOneGroupOut()
    all_preds = []
    all_targets = []
    all_scores = []
    
    indices = np.arange(len(dataset))
    v_dim = dataset[0]['v'].shape[-1]
    t_dim = dataset[0]['t'].shape[-1]
    
    print("\n[Starting LOGO Cross-Validation...]")
    
    for fold, (train_idx, test_idx) in enumerate(logo.split(indices, labels, groups=groups)):
        test_recipe_id = groups[test_idx[0]]
        print(f"--- Fold {fold + 1}/{len(groups)} (Test Recipe: {test_recipe_id}) ---")
        
        # Calculate dynamic positive weight to handle severe class imbalance
        train_labels = [labels[i] for i in train_idx]
        num_pos = sum(train_labels)
        num_neg = len(train_labels) - num_pos
        
        # Apply a soft cap (e.g., maximum 2.0) to prevent the model from collapsing to predicting only 1s
        raw_weight = num_neg / (num_pos + 1e-5)
        soft_weight = min(raw_weight, 4.0)
        pos_weight_val = torch.tensor([soft_weight], dtype=torch.float32).to(device)
        
        train_subdataset = Subset(dataset, train_idx)
        
        # Multiprocessing is disabled because dataset is fully pre-loaded in VRAM
        train_loader = DataLoader(
            train_subdataset, 
            batch_size=batch_size,
            shuffle=True, 
            collate_fn=collate_graph_batch
        )

        # Initialize a fresh model for each fold
        model = GraphClassifier(
            visual_dim=v_dim, 
            text_dim=t_dim, 
            hidden_dim=hidden_dim, 
            num_layers=num_layers, 
            dropout_prob=dropout
        ).to(device)
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_val)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Initialize the Cosine Annealing LR scheduler
        # T_max is set to num_epochs (the total budget of epochs per fold)
        # eta_min defines the final learning rate floor (1e-6 is optimal for deep refinement)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
        
        # Initialize the AMP (Automatic Mixed Precision) Scaler
        scaler = torch.amp.GradScaler('cuda')
        
        # --- Training Phase ---
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            model.train() 
            for batch in train_loader:
                # Setting gradients to None is computationally faster than zeroing them
                optimizer.zero_grad(set_to_none=True)
                
                # Fetch batched tensors (already residing on GPU)
                v_inputs = batch['v']
                t_inputs = batch['t']
                adj_inputs = batch['adj']
                targets = batch['y']
                
                # Mixed Precision context: casts eligible operations to float16
                with torch.amp.autocast('cuda'):
                    logits = model(v_inputs, t_inputs, adj_inputs)
                    loss = criterion(logits, targets)
                
                # Backward pass through the scaler to prevent underflow
                scaler.scale(loss).backward()
                
                # Unscale gradients before clipping to ensure threshold accuracy
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss += loss.item() * len(batch['ids'])
            
            # Update the learning rate at the end of each training epoch
            scheduler.step()
            
            wandb.log({
                "train_loss": epoch_loss / len(train_subdataset),
                "learning_rate": scheduler.get_last_lr()[0],
                "fold": fold + 1,
                "epoch": epoch + 1 + (fold * num_epochs)
            })
            
        # --- Evaluation Phase ---
        model.eval()
        with torch.no_grad():
            test_sample = dataset[test_idx[0]]
            
            # Add the required batch dimension (unsqueeze) for the 3D batched forward pass
            v_test = test_sample['v'].unsqueeze(0)
            t_test = test_sample['t'].unsqueeze(0)
            adj_test = test_sample['adj'].unsqueeze(0)
            
            test_logits = model(v_test, t_test, adj_test)
            prob = torch.sigmoid(test_logits).item()
            
            pred = 1.0 if prob >= th else 0.0
            target = test_sample['y'].item()
            
            all_preds.append(pred)
            all_targets.append(target)
            all_scores.append(prob)
            
            print(f"    Target: {int(target)} | Prediction: {int(pred)} | Correct: {pred == target}")
            
    # Compute and display global cross-validation metrics
    total_metrics = evaluate_metrics(np.array(all_targets), np.array(all_preds), y_scores=np.array(all_scores))
    
    print(f"""
Global GNN (Substep 4) LOGO Results ->
    Acc:   {total_metrics['accuracy']:.4f}
    Prec:  {total_metrics['precision']:.4f}
    Rec:   {total_metrics['recall']:.4f}
    F1:    {total_metrics['f1_score']:.4f}
    AUROC: {total_metrics['auroc']:.4f}""")
    
    best_th, best_f1 = find_optimal_threshold(np.array(all_targets), np.array(all_scores))
    print(f"Optimal Threshold: {best_th:.2f} -> Best F1: {best_f1:.4f}")
    print("\nConfusion Matrix:")
    print(total_metrics['confusion_matrix'])
    wandb.log({
        "global/Accuracy": total_metrics['accuracy'],
        "global/Precision": total_metrics['precision'],
        "global/Recall": total_metrics['recall'],
        "global/F1_Score": total_metrics['f1_score'],
        "global/AUROC": total_metrics['auroc'],
        "global/Optimal_Threshold": best_th,
        "global/Best_F1_at_Threshold": best_f1,
        "global/Probability_Distribution": wandb.Histogram(all_scores),
        "global/Confusion_Matrix": wandb.plot.confusion_matrix(
            y_true=np.array(all_targets), 
            preds=np.array(all_preds), 
            class_names=["Correct (0)", "Error (1)"]
        )
    })
    
    wandb.finish()
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-step Action Graph Neural Network")
    parser.add_argument("--visual_npz", type=str, required=True, help="Path to visual step embeddings")
    parser.add_argument("--text_npz", type=str, required=True, help="Path to text task graph embeddings")
    parser.add_argument("--annotations", type=str, default="annotations/annotation_json/error_annotations.json")
    
    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=15, help="Epochs per fold")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for the DataLoader")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate") 
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for the anomaly detection")
    
    # Architecture and Regularization Hyperparameters
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension size")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of GNN message passing layers")
    parser.add_argument("--dropout", type=float, default=0.4, help="Dropout probability")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="L2 regularization penalty for Adam")
    
    args = parser.parse_args()
    
    wandb.init(
        project="gnn-captain_cook",
        name=f"LOGO_BCE_hd{args.hidden_dim}_ep{args.epochs}",
        config=vars(args)
    )
    
    print("\n[Starting Graph Neural Network Training]")
    print(f"Hardware utilized: {device.type.upper()} (Pre-loading dataset)")
    
    # Load dataset into memory
    dataset, groups, labels = load_graph_data(args.visual_npz, args.text_npz, args.annotations)
    print(f"Dataset loaded: {len(dataset)} graphs.")
    
    # Execute cross-validation
    train_gnn_logo(
        dataset, groups, labels, 
        num_epochs=args.epochs, 
        batch_size=args.batch_size, 
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        th = args.threshold
    )