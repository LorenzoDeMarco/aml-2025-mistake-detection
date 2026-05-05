import os
import json
import torch
import torch.nn as nn
import argparse
import numpy as np
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score
from core.models.task_verification import TaskVerificationTransformer

def load_task_verification_data(npz_file, annotations_file):
    # Load the pre-extracted step-level EgoVLP embeddings
    print("Loading .npz features file...")
    data = np.load(npz_file)
    
    # Load the ground-truth JSON annotations
    print("Loading annotations...")
    with open(annotations_file, 'r') as f:
        error_annotations = json.load(f)
        
    # Build a dictionary mapping each recording_id to a global binary label.
    # Label 1 (Error) if AT LEAST ONE step contains an error. Label 0 (Normal) otherwise.
    recipe_labels = {}
    for recording in error_annotations:
        rec_id = recording['recording_id']
        has_error = any("errors" in step for step in recording.get('step_annotations', []))
        recipe_labels[rec_id] = 1 if has_error else 0
        
    X_list = []
    y_list = []
    groups = []
    
    # Extract tensors from the loaded npz file and match them with global labels
    for rec_id in data.files:
        features = data[rec_id] 
        X_list.append(torch.tensor(features, dtype=torch.float32))
        y_list.append(recipe_labels.get(rec_id, 0)) 
        groups.append(rec_id)
        
    # Pad the varying-length step sequences to create a uniform rectangular tensor block
    X_padded = pad_sequence(X_list, batch_first=True)
    
    # Generate the boolean mask for the Transformer Attention mechanism.
    # Positions where the index is >= the original sequence length are marked as True (padding).
    lengths = torch.tensor([len(x) for x in X_list])
    max_len = X_padded.size(1)
    mask = torch.arange(max_len).expand(len(lengths), max_len) >= lengths.unsqueeze(1)
    
    return X_padded, torch.tensor(y_list), np.array(groups), mask

def train_leave_one_out(X, y, groups, mask, input_dim, num_epochs=15):
    # Initialize the Leave-One-Group-Out cross-validation strategy
    logo = LeaveOneGroupOut()
    all_preds = []
    all_targets = []
    
    indices = np.arange(len(y))
    
    # Iterate over each fold. In each iteration, one specific recipe (group) is held out for testing.
    for fold, (train_idx, test_idx) in enumerate(logo.split(indices, y.numpy(), groups=groups)):
        print(f"--- Fold {fold + 1}/{len(groups)} (Test Recipe: {groups[test_idx[0]]}) ---")
        
        # Split data and masks into Train and Test subsets for the current fold
        X_train, y_train, mask_train = X[train_idx], y[train_idx], mask[train_idx]
        X_test, y_test, mask_test = X[test_idx], y[test_idx], mask[test_idx]
        
        # Initialize model, loss function (Binary Cross Entropy with Logits), and optimizer
        model = TaskVerificationTransformer(input_dim=input_dim)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        # --- Training Loop ---
        model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            logits = model(X_train, mask=mask_train)
            loss = criterion(logits, y_train.float())
            loss.backward()
            optimizer.step()
            
        # --- Evaluation Phase ---
        model.eval()
        with torch.no_grad():
            test_logits = model(X_test, mask=mask_test)
            
            # Apply Sigmoid activation and threshold at 0.5 for binary classification
            preds = torch.sigmoid(test_logits) >= 0.5
            
            # Store predictions and ground truths for global metric calculation
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_test.cpu().numpy())
            
            fold_acc = accuracy_score(y_test.cpu().numpy(), preds.cpu().numpy())
            print(f"Fold Accuracy: {fold_acc:.4f}")
            
    # Calculate the overall accuracy across the entire dataset after the Leave-One-Out procedure
    total_acc = accuracy_score(all_targets, all_preds)
    print(f"\n--> Global Leave-One-Out Accuracy: {total_acc:.4f}")

if __name__ == "__main__":
    # Command line arguments parser
    parser = argparse.ArgumentParser(description="Task Verification Baseline")
    parser.add_argument("--npz_file", type=str, required=True, help="Absolute path to the step embeddings .npz file")
    parser.add_argument("--annotations", type=str, default="annotations/annotation_json/error_annotations.json", help="Relative path to the error annotations JSON file")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs per fold")
    args = parser.parse_args()
    
    print("\n[Starting Task Verification Pipeline]")
    
    # Load dataset and construct feature matrices
    X, y, groups, mask = load_task_verification_data(args.npz_file, args.annotations)
    input_dim = X.shape[-1]
    
    print(f"Successfully loaded {len(y)} recipes. Spatial feature dimension: {input_dim}.")
    
    # Execute the Leave-One-Out cross-validation
    train_leave_one_out(X, y, groups, mask, input_dim, num_epochs=args.epochs)