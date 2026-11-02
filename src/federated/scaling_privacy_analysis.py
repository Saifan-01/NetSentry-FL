"""
NetSentry-FL: Empirical Comparison of Feature Scaling Paradigms in Federated Learning
Compares:
1. Pooled QuantileTransformer (Standard simulation baseline; assumes centralized pool)
2. Local QuantileTransformer (Fitted strictly on each client's private local data; zero sharing)
3. Fixed Parameterless Normalization: log1p(x) + robust median/IQR constant clipping (Zero fitting, 100% private)
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from src.models.ids_net import NetSentryClassifier, get_model_parameters, set_model_parameters
from src.federated.client import NetSentryClient
from src.federated.run_federated_experiment import aggregate_fedavg
from src.federated.partitioners import create_extreme_attack_silos

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def evaluate_pred(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0.0
    }

def run_federated_trial(clients_dict, input_dim, X_test, y_test, num_rounds=15, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    clients = [
        NetSentryClient(cid, clients_dict[cid]["X"], clients_dict[cid]["y"], input_dim)
        for cid in range(len(clients_dict))
    ]
    global_model = NetSentryClassifier(input_dim=input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.1)
    global_params = get_model_parameters(global_model)
    config = {"local_epochs": 2, "batch_size": 128, "lr": 0.001, "proximal_mu": 0.0}
    
    for rnd in range(1, num_rounds + 1):
        updates = []
        for client in clients:
            up, num, _ = client.fit(global_params, config)
            updates.append((up, num, {}))
        global_params = aggregate_fedavg(updates)
        set_model_parameters(global_model, global_params)
        
    global_model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(global_model(torch.from_numpy(X_test))).numpy()
    return evaluate_pred(y_test, probs)

def main():
    import os
    torch.set_num_threads(min(12, os.cpu_count() or 4))
    print("[*] Loading unscaled raw partitions for 5-Seed Scaling Privacy Analysis...", flush=True)
    df_raw = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "real_cicids2017_curated.csv")
    train_meta = pd.read_csv(PROCESSED_DIR / "train_metadata.csv")
    test_meta = pd.read_csv(PROCESSED_DIR / "test_metadata.csv")
    
    label_cols = ["Label", "Binary_Label", "Category"]
    feature_cols = [c for c in df_raw.columns if c not in label_cols]
    
    from sklearn.model_selection import train_test_split
    df_train_val, df_test = train_test_split(df_raw, test_size=0.20, stratify=df_raw["Category"], random_state=42)
    df_train, df_val = train_test_split(df_train_val, test_size=0.125, stratify=df_train_val["Category"], random_state=42)
    
    X_train_raw = df_train[feature_cols].values.astype(np.float64)
    y_train = df_train["Binary_Label"].values.astype(np.float32)
    X_test_raw = df_test[feature_cols].values.astype(np.float64)
    y_test = df_test["Binary_Label"].values.astype(np.float32)
    input_dim = len(feature_cols)
    
    seeds = [42, 101, 777, 1234, 5678]
    
    # Storage for multi-seed runs
    p_runs = []
    loc_runs = []
    f_runs = []
    
    # Fixed transform definition
    def fixed_transform(arr):
        log_arr = np.log1p(np.maximum(0, arr))
        clipped = np.clip(log_arr, 0.0, 20.0)
        return ((clipped / 10.0) - 1.0).astype(np.float32)
        
    X_train_f = fixed_transform(X_train_raw)
    X_test_f = fixed_transform(X_test_raw)
    
    print(f"[*] Running 5 seeds {seeds} across 3 scaling paradigms...", flush=True)
    for s in seeds:
        print(f"\n--- Seed {s} ---", flush=True)
        # 1. Pooled Quantile
        scaler_p = QuantileTransformer(output_distribution="normal", random_state=s, n_quantiles=min(1000, len(X_train_raw)))
        X_train_p = np.clip(scaler_p.fit_transform(X_train_raw), -5.0, 5.0).astype(np.float32)
        X_test_p = np.clip(scaler_p.transform(X_test_raw), -5.0, 5.0).astype(np.float32)
        silos_p = create_extreme_attack_silos(X_train_p, train_meta, seed=s)
        m_p = run_federated_trial(silos_p, input_dim, X_test_p, y_test, num_rounds=15, seed=s)
        p_runs.append(m_p)
        print(f"  [Seed {s}] Pooled Quantile: F1={m_p['f1']:.4f}, Recall={m_p['recall']:.4f}, Precision={m_p['precision']:.4f}", flush=True)
        
        # 2. Local Quantile (zero pooling)
        silos_raw = create_extreme_attack_silos(X_train_raw, train_meta, seed=s)
        silos_local = {}
        scalers_local = {}
        for cid in range(3):
            sc = QuantileTransformer(output_distribution="normal", random_state=s, n_quantiles=min(1000, len(silos_raw[cid]["X"])))
            X_loc = np.clip(sc.fit_transform(silos_raw[cid]["X"]), -5.0, 5.0).astype(np.float32)
            silos_local[cid] = {"X": X_loc, "y": silos_raw[cid]["y"], "categories": silos_raw[cid]["categories"]}
            scalers_local[cid] = sc
        X_test_loc = np.clip(scalers_local[0].transform(X_test_raw), -5.0, 5.0).astype(np.float32)
        m_loc = run_federated_trial(silos_local, input_dim, X_test_loc, y_test, num_rounds=15, seed=s)
        loc_runs.append(m_loc)
        print(f"  [Seed {s}] Local Quantile:  F1={m_loc['f1']:.4f}, Recall={m_loc['recall']:.4f}, Precision={m_loc['precision']:.4f}", flush=True)
        
        # 3. Fixed Parameterless
        silos_f = create_extreme_attack_silos(X_train_f, train_meta, seed=s)
        m_f = run_federated_trial(silos_f, input_dim, X_test_f, y_test, num_rounds=15, seed=s)
        f_runs.append(m_f)
        print(f"  [Seed {s}] Fixed Normalization: F1={m_f['f1']:.4f}, Recall={m_f['recall']:.4f}, Precision={m_f['precision']:.4f}", flush=True)
        
    def summarize(runs_list, name, priv):
        f1_vals = [r["f1"] for r in runs_list]
        rec_vals = [r["recall"] for r in runs_list]
        prec_vals = [r["precision"] for r in runs_list]
        fpr_vals = [r["fpr"] for r in runs_list]
        return {
            "Paradigm": name,
            "Privacy_Level": priv,
            "F1_Mean": float(np.mean(f1_vals)),
            "F1_Std": float(np.std(f1_vals)),
            "Recall_Mean": float(np.mean(rec_vals)),
            "Recall_Std": float(np.std(rec_vals)),
            "Precision_Mean": float(np.mean(prec_vals)),
            "Precision_Std": float(np.std(prec_vals)),
            "FPR_Mean": float(np.mean(fpr_vals)),
            "FPR_Std": float(np.std(fpr_vals)),
            "F1": f"{np.mean(f1_vals):.4f} +/- {np.std(f1_vals):.4f}",
            "Recall": f"{np.mean(rec_vals):.4f} +/- {np.std(rec_vals):.4f}",
            "Precision": f"{np.mean(prec_vals):.4f} +/- {np.std(prec_vals):.4f}",
            "False positive rate": f"{np.mean(fpr_vals):.4f} +/- {np.std(fpr_vals):.4f}"
        }
        
    results = [
        summarize(p_runs, "Pooled QuantileTransformer (Simulation Baseline)", "Low (Assumes pooled data)"),
        summarize(loc_runs, "Local QuantileTransformer (Strictly Decentralized)", "High (Zero raw data or quantile pooling)"),
        summarize(f_runs, "Fixed Parameterless Normalization (log1p + Constant Bounds)", "Complete (Zero fitting, 100% static)")
    ]
    
    df_scal = pd.DataFrame(results)
    df_scal.to_csv(EXPERIMENTS_DIR / "scaling_privacy_comparison.csv", index=False)
    print("\nSaved 5-seed scaling comparison to experiments/scaling_privacy_comparison.csv")
    print(df_scal[["Paradigm", "Recall", "F1", "Precision", "False positive rate"]].to_string(index=False))

if __name__ == "__main__":
    main()
