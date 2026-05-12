import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import warnings
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score
from core.models.graph_model import GraphClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            
            # Creation of the adjacency matrix (Identity + Sequential Edges)
            # Add self-loops (the diagonal) to prevent a node from forgetting itself during message passing
            adj = torch.eye(num_nodes).to(device)
            for i in range(num_nodes - 1):
                adj[i, i+1] = 1.0 
            
            y = recipe_labels.get(rec_id, 0)
            
            dataset.append({
                'id': rec_id,
                'v': v_feats,
                't': t_feats,
                'adj': adj,
                'y': torch.tensor(y, dtype=torch.float32).to(device)
            })
            groups.append(rec_id)
            labels.append(y)
            
    return dataset, np.array(groups), np.array(labels)

def train_gnn_logo(dataset, groups, labels, num_epochs=15):
    logo = LeaveOneGroupOut()
    all_preds = []
    all_targets = []
    
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
        
        # Initialize a fresh model for each fold
        model = GraphClassifier(visual_dim=v_dim, text_dim=t_dim, hidden_dim=256, num_layers=2).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_val)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        # Training
        model.train()
        for epoch in range(num_epochs):
            # Shuffle training indices at each epoch for robustness
            np.random.shuffle(train_idx)
            
            epoch_loss = 0.0
            optimizer.zero_grad()
            
            for idx in train_idx:
                sample = dataset[idx]
                
                # Forward pass of the single graph
                logits = model(sample['v'], sample['t'], sample['adj'])
                loss = criterion(logits, sample['y'])
                
                scaled_loss = loss/len(train_idx)
                scaled_loss.backward()
                
                epoch_loss += loss.item()
                
            # Update weights once the entire epoch is passed (Full-batch Gradient Descent)
            # This stabilizes learning on small datasets
            optimizer.step()
            
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_sample = dataset[test_idx[0]]
            test_logits = model(test_sample['v'], test_sample['t'], test_sample['adj'])
            
            pred = (torch.sigmoid(test_logits) >= 0.5).float().item()
            target = test_sample['y'].item()
            
            all_preds.append(pred)
            all_targets.append(target)
            
            res_str = "CORRECT" if pred == target else "WRONG"
            print(f"Pred: {pred} | True: {target} [{res_str}]")
            
    # Final accuracy
    total_acc = accuracy_score(all_targets, all_preds)
    print(f"--> Global GNN (Substep 4) Accuracy: {total_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-step Action Graph Neural Network")
    parser.add_argument("--visual_npz", type=str, required=True, help="Path to visual step embeddings")
    parser.add_argument("--text_npz", type=str, required=True, help="Path to text task graph embeddings")
    parser.add_argument("--annotations", type=str, default="annotations/annotation_json/error_annotations.json")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs per fold")
    args = parser.parse_args()
    
    print("\n[Starting Graph Neural Network Training]")
    print(f"Hardware utilized: {device.type.upper()}")
    
    dataset, groups, labels = load_graph_data(args.visual_npz, args.text_npz, args.annotations)
    print(f"Dataset loaded: {len(dataset)} graphs.")
    
    train_gnn_logo(dataset, groups, labels, num_epochs=args.epochs)