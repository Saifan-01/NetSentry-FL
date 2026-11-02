"""
NetSentry-FL: Centralized Baseline Classifiers on Real CIC-IDS2017
Trains and evaluates conventional ML and Deep Learning models on the 70/10/20 split:
1. Logistic Regression (Linear baseline)
2. Random Forest (Ensemble non-linear baseline)
3. Centralized PyTorch MLP (Deep Learning baseline with validation-based early stopping)
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix
)
from typing import Dict, Any, List

from src.models.ids_net import NetSentryClassifier

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

def load_processed_splits():
    """Loads 70/10/20 preprocessed train, val, and test arrays."""
    X_train = np.load(PROCESSED_DIR / "X_train.npy").astype(np.float32)
    y_train = np.load(PROCESSED_DIR / "y_train.npy").astype(np.float32)
    X_val = np.load(PROCESSED_DIR / "X_val.npy").astype(np.float32)
    y_val = np.load(PROCESSED_DIR / "y_val.npy").astype(np.float32)
    X_test = np.load(PROCESSED_DIR / "X_test.npy").astype(np.float32)
    y_test = np.load(PROCESSED_DIR / "y_test.npy").astype(np.float32)
    feature_names = joblib.load(PROCESSED_DIR / "feature_names.pkl")
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_names

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, model_name: str) -> Dict[str, Any]:
    """Computes comprehensive intrusion detection evaluation metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc = roc_auc_score(y_true, y_prob)
    
    p_curve, r_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(r_curve, p_curve)
    cm = confusion_matrix(y_true, y_pred)
    
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print(f"\n========================================================")
    print(f"  Evaluation Results: {model_name}")
    print(f"========================================================")
    print(f"  Accuracy:          {acc * 100:.2f}%")
    print(f"  Precision (PPV):   {prec * 100:.2f}% (Attack precision)")
    print(f"  Recall (TPR):      {rec * 100:.2f}% (Attack detection rate)")
    print(f"  F1-Score:          {f1:.4f}")
    print(f"  ROC-AUC:           {roc:.4f}")
    print(f"  PR-AUC:            {pr_auc:.4f}")
    print(f"  False Alarm (FPR): {fpr * 100:.2f}%")
    print(f"  Confusion Matrix:  TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"========================================================\n")
    
    return {
        "model": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc, 4),
        "pr_auc": round(pr_auc, 4),
        "fpr": round(fpr, 4),
        "confusion_matrix": f"[[{tn}, {fp}], [{fn}, {tp}]]"
    }

def train_and_eval_baselines(random_state: int = 42):
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = load_processed_splits()
    results = []

    # 1. Logistic Regression Baseline
    print("[*] Training Logistic Regression baseline (max_iter=1000)...")
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]
    results.append(evaluate_predictions(y_test, y_pred_lr, y_prob_lr, "Logistic Regression"))

    # 2. Random Forest Baseline (Centralized Non-Linear Upper Bound)
    print("[*] Training Random Forest baseline (100 estimators)...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=16, n_jobs=-1, random_state=random_state)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    results.append(evaluate_predictions(y_test, y_pred_rf, y_prob_rf, "Random Forest (Upper Bound)"))

    # 3. Centralized PyTorch MLP Baseline (Federated-Compatible Architecture)
    print("[*] Training Centralized PyTorch MLP baseline with validation monitoring...")
    torch.manual_seed(random_state)
    input_dim = X_train.shape[1]
    mlp = NetSentryClassifier(input_dim=input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.1)
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=0.001, weight_decay=1e-4)
    
    pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / (y_train.sum() + 1e-5)])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=128,
        shuffle=True
    )
    
    best_val_f1 = 0.0
    best_model_weights = None
    
    for epoch in range(1, 26):
        mlp.train()
        total_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = mlp(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Validation check
        mlp.eval()
        with torch.no_grad():
            val_logits = mlp(torch.from_numpy(X_val))
            val_probs = torch.sigmoid(val_logits).numpy()
            val_preds = (val_probs >= 0.5).astype(int)
            val_f1 = f1_score(y_val, val_preds, zero_division=0)
            
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_weights = mlp.state_dict().copy()
            
        if epoch % 5 == 0 or epoch == 25:
            print(f"    Epoch {epoch:2d}/25 - Loss: {total_loss/len(train_loader):.4f} - Val F1: {val_f1:.4f} (Best: {best_val_f1:.4f})")
            
    # Load best weights
    if best_model_weights is not None:
        mlp.load_state_dict(best_model_weights)
        
    mlp.eval()
    with torch.no_grad():
        test_logits = mlp(torch.from_numpy(X_test))
        test_probs = torch.sigmoid(test_logits).numpy()
        test_preds = (test_probs >= 0.5).astype(int)
        
    results.append(evaluate_predictions(y_test, test_preds, test_probs, "Centralized PyTorch MLP"))
    
    # Save results to CSV
    res_df = pd.DataFrame(results)
    out_csv = EXPERIMENTS_DIR / "centralized_baseline_results.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"[+] Saved genuine centralized baseline results to {out_csv}")
    
    # Save model weights for centralized upper bound
    torch.save(mlp.state_dict(), EXPERIMENTS_DIR / "centralized_model.pth")
    return results

if __name__ == "__main__":
    train_and_eval_baselines()
