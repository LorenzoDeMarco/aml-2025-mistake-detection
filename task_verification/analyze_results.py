import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, 
                             roc_curve, roc_auc_score, precision_score, recall_score)

def analyze(csv_path):
    df = pd.read_csv(csv_path)
    y_true = df['ground_truth'].values
    y_prob = df['probability'].values
    
    # optimal threshold using Youden's J statistic
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    J = tpr - fpr
    best_idx = np.argmax(J)
    best_thresh = thresholds[best_idx]

    # metrics with standard threshold 
    y_pred_std = (y_prob >= 0.5).astype(int)
    
    # calculate metrics with optimal threshold
    y_pred_opt = (y_prob >= best_thresh).astype(int)
    
    opt_acc = accuracy_score(y_true, y_pred_opt)
    opt_prec = precision_score(y_true, y_pred_opt, zero_division=0)
    opt_rec = recall_score(y_true, y_pred_opt, zero_division=0)
    opt_f1 = f1_score(y_true, y_pred_opt, zero_division=0)
    
    # Output Console
    print("\n================ FINAL GLOBAL GNN PERFORMANCE EVALUATION (LOGO) ================")
    print(f"wandb: Final_AUROC {roc_auc_score(y_true, y_prob):.5f}")
    print(f"--- Standard Threshold (0.5) ---")
    print(f"Acc: {accuracy_score(y_true, y_pred_std):.4f} | F1: {f1_score(y_true, y_pred_std):.4f}")
    print(f"--- Ricalibration with optimal threshold (Thresh={best_thresh:.4f}) ---")
    print(f"wandb:   Opt_Acc  {opt_acc:.5f}")
    print(f"wandb:   Opt_Prec {opt_prec:.5f}")
    print(f"wandb:   Opt_Rec  {opt_rec:.5f}")
    print(f"wandb:   Opt_F1   {opt_f1:.5f}")
    print("===============================================================================")
    
    cm = confusion_matrix(y_true, y_pred_opt)
    
    plt.figure(figsize=(12, 5))
    
    # Plot confusion matrix
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Correct', 'Error'], yticklabels=['Correct', 'Error'])
    plt.title(f'Confusion Matrix (Thresh: {best_thresh:.2f})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Plot ROC Curve
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, label=f'GNN (AUROC={roc_auc_score(y_true, y_prob):.2f})', color='blue', lw=2)
    plt.plot([0,1], [0,1], 'k--', lw=1)
    plt.scatter(fpr[best_idx], tpr[best_idx], marker='o', color='red', label='Best Threshold')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze("logo_gnn_error_analysis.csv")