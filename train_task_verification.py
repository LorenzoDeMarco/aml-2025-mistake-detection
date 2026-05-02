import torch
import torch.nn as nn
import argparse
from core.models.task_verification import TaskVerificationTransformer
import torch.optim as optim
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score
import numpy as np

def train_leave_one_out(X, y, groups, input_dim, num_epochs=10):
    logo = LeaveOneGroupOut()
    
    all_preds=[]
    all_targets=[]
    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups)):
        print(f"--- Fold {fold + 1} (Test Recipe: {groups[test_idx[0]]}) ---")
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        model = TaskVerificationTransformer(input_dim=input_dim)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        model.train()
        for _epoch in range(num_epochs):
            optimizer.zero_grad()
            logits = model(X_train)
            loss = criterion(logits, y_train.float())
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            test_logits = model(X_test)
            preds = torch.sigmoid(test_logits) >= 0.5
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_test.cpu().numpy())
            
            fold_acc = accuracy_score(all_targets, all_preds)
            print(f"Accuracy: {fold_acc:.4f}")
            
        total_acc = accuracy_score(all_targets, all_preds)
        print(f"--> Gloabl Leave-One-Out Accuracy: {total_acc:.4f}")
        