"""
NetSentry-FL: Phase 2 - Baseline Centralized IDS Models
Trains and evaluates conventional ML models:
1. Logistic Regression (Linear baseline)
2. Random Forest (Ensemble baseline)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from typing import Dict, Any

PROCESSED_DIR = Path("data/processed")
EXPERIMENTS_DIR = Path("experiments")

def load_processed_splits():
    """Loads preprocessed train, val, and test arrays."""
    X_train = np.load(PROCESSED_DIR / "X_train.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    X_val = np.load(PROCESSED_DIR / "X_val.npy")
    y_val = np.load(PROCESSED_DIR / "y_val.npy")
    X_test = np.load(PROCESSED_DIR / "X_test.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")
    feature_names = joblib.load(PROCESSED_DIR / "feature_names.pkl")
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_names

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, model_name: str) -> Dict[str, Any]:
    """Computes standard intrusion detection metrics."""
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
    print(f"  Confusion Matrix:\n    [TN={tn}, FP={fp}]\n    [FN={fn}, TP={tp}]")
    print(f"========================================================\n")
    
    return {
        "model": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc,
        "pr_auc": pr_auc,
        "fpr": fpr,
        "confusion_matrix": cm.tolist()
    }

def train_and_eval_baselines():
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = load_processed_splits()
    
    results = []

    # 1. Logistic Regression Baseline
    print("[*] Training Logistic Regression baseline...")
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]
    results.append(evaluate_predictions(y_test, y_pred_lr, y_prob_lr, "Logistic Regression"))

    # 2. Random Forest Baseline
    print("[*] Training Random Forest baseline (100 estimators)...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=16, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    results.append(evaluate_predictions(y_test, y_pred_rf, y_prob_rf, "Random Forest"))

    # Save summary results
    res_df = pd.DataFrame(results)
    res_df.to_csv(EXPERIMENTS_DIR / "centralized_baseline_results.csv", index=False)
    print(f"[+] Saved baseline comparison to {EXPERIMENTS_DIR / 'centralized_baseline_results.csv'}")

    # Top feature importances from Random Forest
    importances = rf.feature_importances_
    top_indices = np.argsort(importances)[::-1][:10]
    print("\n[*] Top 10 Most Predictive Intrusion Detection Features (Random Forest):")
    for rank, idx in enumerate(top_indices, 1):
        print(f"    {rank:2d}. {feature_names[idx]:<30} (Importance: {importances[idx]:.4f})")

if __name__ == "__main__":
    train_and_eval_baselines()
