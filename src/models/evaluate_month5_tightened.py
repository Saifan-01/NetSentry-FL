"""
NetSentry-FL: Tightened Month 5 Evaluation Harness (Zero Test-Leakage & 5 Seeds)
Incorporates:
1. Validation-Set-Based Hyperparameter Tuning for FedProx (mu in {0.001, 0.01, 0.1, 1.0}, E in {1, 2, 5})
2. Web Attack Minority Class Collapse Ablation (Sample-Weighted vs. Equal-Weighted 1/K vs. FedProx)
3. 5-Seed Evaluation on 7,292 Held-Out Test Flows (Seeds: 42, 101, 777, 1234, 5678)
4. Per-Attack-Type Recall (DoS, PortScan, Web Attack) Mean +/- Std across all 5 seeds
5. Final-Round Reporting (Strict R=15, zero round-selection test leakage)
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
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, f1_score, precision_score, recall_score
import joblib

from src.models.ids_net import NetSentryClassifier, get_model_parameters, set_model_parameters
from src.federated.client import NetSentryClient
from src.federated.partitioners import (
    create_extreme_attack_silos,
    create_dirichlet_partitions,
    plot_client_class_distributions
)

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHARTS_DIR = PROJECT_ROOT / "reports" / "charts"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

import os
torch.set_num_threads(min(12, os.cpu_count() or 4))

def fast_eval(y_true: np.ndarray, y_prob: np.ndarray):
    y_pred = (y_prob >= 0.5)
    tp = np.sum(y_pred & (y_true == 1))
    fp = np.sum(y_pred & (y_true == 0))
    fn = np.sum((~y_pred) & (y_true == 1))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"precision": float(prec), "recall": float(rec), "f1": float(f1)}

def evaluate_metrics(y_true: np.ndarray, y_prob: np.ndarray):
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    p_curve, r_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(r_curve, p_curve)
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(pr_auc),
        "fpr": float(fpr),
        "cm": cm
    }

def aggregate_weights(updates, weighting_type="sample_weighted"):
    if weighting_type == "equal_weighted":
        weights = [1.0 / len(updates)] * len(updates)
    else:
        total_samples = sum(num for _, num, _ in updates)
        weights = [num / total_samples for _, num, _ in updates]
        
    num_layers = len(updates[0][0])
    avg_params = []
    for layer_idx in range(num_layers):
        layer_avg = sum(w * up[layer_idx] for (up, _, _), w in zip(updates, weights))
        avg_params.append(layer_avg)
    return avg_params

def run_fl_simulation(
    clients_dict: dict,
    input_dim: int,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_rounds: int = 15,
    local_epochs: int = 2,
    lr: float = 0.001,
    proximal_mu: float = 0.0,
    weighting_type: str = "sample_weighted",
    setup_name: str = "FL",
    seed: int = 42
):
    clients = [
        NetSentryClient(cid, clients_dict[cid]["X"], clients_dict[cid]["y"], input_dim)
        for cid in range(len(clients_dict))
    ]
    global_model = NetSentryClassifier(input_dim=input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.1)
    global_params = get_model_parameters(global_model)
    
    X_val_t = torch.from_numpy(X_val)
    X_test_t = torch.from_numpy(X_test)
    
    round_history = []
    config = {
        "local_epochs": local_epochs,
        "batch_size": 128,
        "lr": lr,
        "proximal_mu": proximal_mu
    }
    
    for rnd in range(1, num_rounds + 1):
        updates = []
        for client in clients:
            up, num, _ = client.fit(global_params, config)
            updates.append((up, num, {}))
            
        global_params = aggregate_weights(updates, weighting_type=weighting_type)
        set_model_parameters(global_model, global_params)
        
        # Validation / Test evaluation at each round
        global_model.eval()
        with torch.no_grad():
            v_probs = torch.sigmoid(global_model(X_val_t)).numpy()
            t_probs = torch.sigmoid(global_model(X_test_t)).numpy()
            
        if rnd == num_rounds:
            v_m = evaluate_metrics(y_val, v_probs)
            t_m = evaluate_metrics(y_test, t_probs)
        else:
            v_m = fast_eval(y_val, v_probs)
            t_m = fast_eval(y_test, t_probs)
        
        round_history.append({
            "seed": seed,
            "round": rnd,
            "setup": setup_name,
            "val_f1": v_m["f1"],
            "test_f1": t_m["f1"],
            "recall": t_m["recall"],
            "precision": t_m["precision"]
        })
        
    return global_model, t_probs, v_probs, round_history

def main():
    seeds = [42, 101, 777, 1234, 5678]
    num_rounds = 15
    
    print("="*95, flush=True)
    print("      NETSENTRY-FL: TIGHTENED MONTH 5 BENCHMARK (VALIDATION TUNING + 5 SEEDS)      ", flush=True)
    print("="*95, flush=True)
    
    # 1. Load Data
    X_train = np.load(PROCESSED_DIR / "X_train.npy").astype(np.float32)
    y_train = np.load(PROCESSED_DIR / "y_train.npy").astype(np.float32)
    X_val = np.load(PROCESSED_DIR / "X_val.npy").astype(np.float32)
    y_val = np.load(PROCESSED_DIR / "y_val.npy").astype(np.float32)
    X_test = np.load(PROCESSED_DIR / "X_test.npy").astype(np.float32)
    y_test = np.load(PROCESSED_DIR / "y_test.npy").astype(np.float32)
    
    train_meta = pd.read_csv(PROCESSED_DIR / "train_metadata.csv")
    val_meta = pd.read_csv(PROCESSED_DIR / "val_metadata.csv")
    test_meta = pd.read_csv(PROCESSED_DIR / "test_metadata.csv")
    input_dim = X_train.shape[1]
    
    dummy_model = NetSentryClassifier(input_dim=input_dim)
    param_bytes = sum(p.numel() * 4 for p in dummy_model.parameters())
    print(f"[*] Input Features: {input_dim} | Model Footprint: {param_bytes / 1024:.2f} KB | 5 Seeds: {seeds}", flush=True)
    
    # -------------------------------------------------------------
    # STEP A: VALIDATION-BASED HYPERPARAMETER SELECTION (Zero Test Leakage)
    # -------------------------------------------------------------
    print("\n>>> STEP A: Hyperparameter Tuning on Validation Split (3,646 flows) <<<", flush=True)
    silos_val = create_extreme_attack_silos(X_train, train_meta, seed=42)
    dir01_val = create_dirichlet_partitions(X_train, train_meta, alpha=0.1, seed=42)
    
    val_csv_path = EXPERIMENTS_DIR / "validation_tuning_grid.csv"
    if val_csv_path.exists():
        print(f"[*] Loading existing validation tuning results from {val_csv_path} (Zero Test Leakage)...", flush=True)
        df_val_tuning = pd.read_csv(val_csv_path)
        silo_rows = df_val_tuning[df_val_tuning["Partition"] == "Extreme Silos"]
        best_silo_row = silo_rows.loc[silo_rows["Val_F1"].idxmax()]
        best_silo_mu = float(best_silo_row["Mu"])
        best_silo_ep = int(best_silo_row["Epochs"])
        best_silo_val_f1 = float(best_silo_row["Val_F1"])
        
        d01_rows = df_val_tuning[df_val_tuning["Partition"] == "Dirichlet alpha=0.1"]
        best_d01_row = d01_rows.loc[d01_rows["Val_F1"].idxmax()]
        best_d01_mu = float(best_d01_row["Mu"])
        best_d01_ep = int(best_d01_row["Epochs"])
        best_d01_val_f1 = float(best_d01_row["Val_F1"])
    else:
        mu_grid = [0.001, 0.01, 0.1, 1.0]
        epoch_grid = [1, 2, 5]
        val_records = []
        
        best_silo_val_f1 = -1.0
        best_silo_mu = 0.01
        best_silo_ep = 2
        
        best_d01_val_f1 = -1.0
        best_d01_mu = 0.01
        best_d01_ep = 2
        
        # Validation grid search on Silos
        print("  [Validation Tuning: Extreme Silos]", flush=True)
        for ep in epoch_grid:
            for mu in mu_grid:
                _, _, v_probs, _ = run_fl_simulation(
                    silos_val, input_dim, X_val, y_val, X_test, y_test, num_rounds=10,
                    local_epochs=ep, lr=0.001, proximal_mu=mu, setup_name="Val_Tuning_Silos", seed=42
                )
                v_m = evaluate_metrics(y_val, v_probs)
                val_records.append({
                    "Partition": "Extreme Silos", "Epochs": ep, "Mu": mu,
                    "Val_F1": round(v_m["f1"], 4), "Val_Recall": round(v_m["recall"], 4), "Val_FPR": round(v_m["fpr"], 4)
                })
                if v_m["f1"] > best_silo_val_f1:
                    best_silo_val_f1 = v_m["f1"]
                    best_silo_mu = mu
                    best_silo_ep = ep
                    
        # Validation grid search on Dirichlet 0.1
        print("  [Validation Tuning: Dirichlet alpha=0.1]", flush=True)
        for ep in [2]:
            for mu in mu_grid:
                _, _, v_probs, _ = run_fl_simulation(
                    dir01_val, input_dim, X_val, y_val, X_test, y_test, num_rounds=10,
                    local_epochs=ep, lr=0.001, proximal_mu=mu, setup_name="Val_Tuning_Dir01", seed=42
                )
                v_m = evaluate_metrics(y_val, v_probs)
                val_records.append({
                    "Partition": "Dirichlet alpha=0.1", "Epochs": ep, "Mu": mu,
                    "Val_F1": round(v_m["f1"], 4), "Val_Recall": round(v_m["recall"], 4), "Val_FPR": round(v_m["fpr"], 4)
                })
                if v_m["f1"] > best_d01_val_f1:
                    best_d01_val_f1 = v_m["f1"]
                    best_d01_mu = mu
                    best_d01_ep = ep
                    
        df_val_tuning = pd.DataFrame(val_records)
        df_val_tuning.to_csv(val_csv_path, index=False)
        
    print(f"[+] Validation Selection Complete (Zero Test Leakage):")
    print(f"    - Extreme Silos: Selected mu* = {best_silo_mu}, Local Epochs E* = {best_silo_ep} (Val F1: {best_silo_val_f1:.4f})")
    print(f"    - Dirichlet 0.1: Selected mu* = {best_d01_mu}, Local Epochs E* = {best_d01_ep} (Val F1: {best_d01_val_f1:.4f})")
    
    # -------------------------------------------------------------
    # STEP B: 5-SEED EVALUATION ON HELD-OUT TEST PARTITION (7,292 flows)
    # -------------------------------------------------------------
    print(f"\n>>> STEP B: 5-Seed Evaluation on 7,292 Held-Out Test Flows (Strict Final Round R={num_rounds}) <<<", flush=True)
    
    setups = [
        "Centralized (MLP)",
        "Local-only (Client A - DoS)",
        "Local-only (Client B - PortScan)",
        "Local-only (Client C - Web Attack)",
        "Federated, IID (FedAvg)",
        "Federated, non-IID Silos (FedAvg)",
        "Federated, non-IID Silos (FedAvg Equal-Weighted 1/K)",
        f"Federated, non-IID Silos (FedProx mu*={best_silo_mu})",
        "Federated, non-IID Dir(0.5) (FedAvg)",
        "Federated, non-IID Dir(0.1) (FedAvg)",
        f"Federated, non-IID Dir(0.1) (FedProx mu*={best_d01_mu})"
    ]
    all_runs = {s: [] for s in setups}
    all_convergence_records = []
    per_attack_runs = []
    
    last_global_model = None

    for seed in seeds:
        print(f"\n--- Running Seed {seed} ---", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # 1. Centralized MLP Baseline
        cent_model = NetSentryClassifier(input_dim=input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.1)
        opt_cent = torch.optim.AdamW(cent_model.parameters(), lr=0.001, weight_decay=1e-4)
        pos_wt_val = min((len(y_train) - y_train.sum()) / (y_train.sum() + 1e-5), 10.0)
        crit_cent = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_wt_val]))
        loader_cent = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=128, shuffle=True, drop_last=True)
        
        cent_model.train()
        for epoch in range(15):
            for bx, by in loader_cent:
                if bx.size(0) <= 1:
                    continue
                opt_cent.zero_grad()
                crit_cent(cent_model(bx), by).backward()
                opt_cent.step()
                
        cent_model.eval()
        with torch.no_grad():
            cent_probs = torch.sigmoid(cent_model(torch.from_numpy(X_test))).numpy()
        cent_m = evaluate_metrics(y_test, cent_probs)
        all_runs["Centralized (MLP)"].append(cent_m)
        print(f"  Centralized MLP: F1={cent_m['f1']:.4f}, Recall={cent_m['recall']:.4f}", flush=True)
        
        # 2. Local-Only Models (Extreme Silos)
        extreme_silos = create_extreme_attack_silos(X_train, train_meta, seed=seed)
        local_probs_dict = {}
        silo_names = [
            "Local-only (Client A - DoS)",
            "Local-only (Client B - PortScan)",
            "Local-only (Client C - Web Attack)"
        ]
        for cid, sname in enumerate(silo_names):
            l_model = NetSentryClassifier(input_dim=input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.1)
            l_opt = torch.optim.AdamW(l_model.parameters(), lr=0.001, weight_decay=1e-4)
            pos_ct = extreme_silos[cid]["y"].sum()
            neg_ct = len(extreme_silos[cid]["y"]) - pos_ct
            l_pos_wt = torch.tensor([min(neg_ct / max(pos_ct, 1.0), 10.0)])
            l_crit = nn.BCEWithLogitsLoss(pos_weight=l_pos_wt)
            
            l_loader = DataLoader(
                TensorDataset(torch.from_numpy(extreme_silos[cid]["X"]), torch.from_numpy(extreme_silos[cid]["y"])),
                batch_size=128, shuffle=True, drop_last=True
            )
            l_model.train()
            for _ in range(8):
                for bx, by in l_loader:
                    if bx.size(0) <= 1:
                        continue
                    l_opt.zero_grad()
                    l_crit(l_model(bx), by).backward()
                    l_opt.step()
                    
            l_model.eval()
            with torch.no_grad():
                l_probs = torch.sigmoid(l_model(torch.from_numpy(X_test))).numpy()
            local_probs_dict[cid] = l_probs
            lm = evaluate_metrics(y_test, l_probs)
            all_runs[sname].append(lm)
            
        # 3. Federated IID (FedAvg)
        idx_perm = np.random.permutation(len(X_train))
        iid_splits = np.array_split(idx_perm, 3)
        iid_clients_dict = {
            cid: {"X": X_train[iid_splits[cid]], "y": y_train[iid_splits[cid]]}
            for cid in range(3)
        }
        fed_iid_model, f_iid_probs, _, hist_iid = run_fl_simulation(
            iid_clients_dict, input_dim, X_val, y_val, X_test, y_test, num_rounds=num_rounds,
            local_epochs=2, lr=0.001, proximal_mu=0.0, weighting_type="sample_weighted",
            setup_name="Federated, IID (FedAvg)", seed=seed
        )
        last_global_model = fed_iid_model
        m_iid = evaluate_metrics(y_test, f_iid_probs)
        all_runs["Federated, IID (FedAvg)"].append(m_iid)
        all_convergence_records.extend(hist_iid)
        print(f"    [Seed {seed}] FedAvg (IID): F1={m_iid['f1']:.4f}, Recall={m_iid['recall']:.4f}", flush=True)
        
        # 4. Federated Non-IID Silos (FedAvg, Sample-Weighted)
        fed_silo_model, f_silo_probs, _, hist_silo = run_fl_simulation(
            extreme_silos, input_dim, X_val, y_val, X_test, y_test, num_rounds=num_rounds,
            local_epochs=2, lr=0.001, proximal_mu=0.0, weighting_type="sample_weighted",
            setup_name="Federated, non-IID Silos (FedAvg)", seed=seed
        )
        m_silo = evaluate_metrics(y_test, f_silo_probs)
        all_runs["Federated, non-IID Silos (FedAvg)"].append(m_silo)
        all_convergence_records.extend(hist_silo)
        print(f"    [Seed {seed}] FedAvg (Silos): F1={m_silo['f1']:.4f}, Recall={m_silo['recall']:.4f}", flush=True)
        
        # 5. Federated Non-IID Silos (FedAvg, Equal-Weighted 1/K - Web Attack Ablation 1)
        _, f_silo_eq_probs, _, hist_silo_eq = run_fl_simulation(
            extreme_silos, input_dim, X_val, y_val, X_test, y_test, num_rounds=num_rounds,
            local_epochs=2, lr=0.001, proximal_mu=0.0, weighting_type="equal_weighted",
            setup_name="Federated, non-IID Silos (FedAvg Equal-Weighted 1/K)", seed=seed
        )
        m_eq = evaluate_metrics(y_test, f_silo_eq_probs)
        all_runs["Federated, non-IID Silos (FedAvg Equal-Weighted 1/K)"].append(m_eq)
        all_convergence_records.extend(hist_silo_eq)
        print(f"    [Seed {seed}] FedAvg (Equal 1/K Silos): F1={m_eq['f1']:.4f}, Recall={m_eq['recall']:.4f}", flush=True)
        
        # 6. Federated Non-IID Silos (FedProx with validation-selected mu*)
        prox_silo_name = f"Federated, non-IID Silos (FedProx mu*={best_silo_mu})"
        _, f_prox_silo_probs, _, hist_prox_silo = run_fl_simulation(
            extreme_silos, input_dim, X_val, y_val, X_test, y_test, num_rounds=num_rounds,
            local_epochs=best_silo_ep, lr=0.001, proximal_mu=best_silo_mu, weighting_type="sample_weighted",
            setup_name=prox_silo_name, seed=seed
        )
        m_prox_silo = evaluate_metrics(y_test, f_prox_silo_probs)
        all_runs[prox_silo_name].append(m_prox_silo)
        all_convergence_records.extend(hist_prox_silo)
        print(f"    [Seed {seed}] FedProx (mu*={best_silo_mu} Silos): F1={m_prox_silo['f1']:.4f}, Recall={m_prox_silo['recall']:.4f}", flush=True)
        
        # 7. Federated Non-IID Dirichlet alpha=0.5 (FedAvg)
        dir_05_silos = create_dirichlet_partitions(X_train, train_meta, alpha=0.5, seed=seed)
        _, f_d05_probs, _, hist_d05 = run_fl_simulation(
            dir_05_silos, input_dim, X_val, y_val, X_test, y_test, num_rounds=num_rounds,
            local_epochs=2, lr=0.001, proximal_mu=0.0, weighting_type="sample_weighted",
            setup_name="Federated, non-IID Dir(0.5) (FedAvg)", seed=seed
        )
        m_d05 = evaluate_metrics(y_test, f_d05_probs)
        all_runs["Federated, non-IID Dir(0.5) (FedAvg)"].append(m_d05)
        all_convergence_records.extend(hist_d05)
        print(f"    [Seed {seed}] FedAvg (Dir 0.5): F1={m_d05['f1']:.4f}, Recall={m_d05['recall']:.4f}", flush=True)
        
        # 8. Federated Non-IID Dirichlet alpha=0.1 (FedAvg)
        dir_01_silos = create_dirichlet_partitions(X_train, train_meta, alpha=0.1, seed=seed)
        _, f_d01_probs, _, hist_d01 = run_fl_simulation(
            dir_01_silos, input_dim, X_val, y_val, X_test, y_test, num_rounds=num_rounds,
            local_epochs=2, lr=0.001, proximal_mu=0.0, weighting_type="sample_weighted",
            setup_name="Federated, non-IID Dir(0.1) (FedAvg)", seed=seed
        )
        m_d01 = evaluate_metrics(y_test, f_d01_probs)
        all_runs["Federated, non-IID Dir(0.1) (FedAvg)"].append(m_d01)
        all_convergence_records.extend(hist_d01)
        print(f"    [Seed {seed}] FedAvg (Dir 0.1): F1={m_d01['f1']:.4f}, Recall={m_d01['recall']:.4f}", flush=True)
        
        # 9. Federated Non-IID Dirichlet alpha=0.1 (FedProx with validation-selected mu*)
        prox_d01_name = f"Federated, non-IID Dir(0.1) (FedProx mu*={best_d01_mu})"
        _, f_prox_d01_probs, _, hist_prox_d01 = run_fl_simulation(
            dir_01_silos, input_dim, X_val, y_val, X_test, y_test, num_rounds=num_rounds,
            local_epochs=best_d01_ep, lr=0.001, proximal_mu=best_d01_mu, weighting_type="sample_weighted",
            setup_name=prox_d01_name, seed=seed
        )
        m_prox_d01 = evaluate_metrics(y_test, f_prox_d01_probs)
        all_runs[prox_d01_name].append(m_prox_d01)
        all_convergence_records.extend(hist_prox_d01)
        print(f"    [Seed {seed}] FedProx (mu*={best_d01_mu} Dir 0.1): F1={m_prox_d01['f1']:.4f}, Recall={m_prox_d01['recall']:.4f}", flush=True)
        
        # Record Per-Attack-Type Recall across all 5 seeds
        preds_eval_dict = {
            "Centralized (MLP)": (cent_probs >= 0.5).astype(int),
            "Local Client A (DoS)": (local_probs_dict[0] >= 0.5).astype(int),
            "Local Client B (PortScan)": (local_probs_dict[1] >= 0.5).astype(int),
            "Local Client C (Web Attack)": (local_probs_dict[2] >= 0.5).astype(int),
            "Federated IID (FedAvg)": (f_iid_probs >= 0.5).astype(int),
            "Federated Silos (FedAvg)": (f_silo_probs >= 0.5).astype(int),
            "Federated Silos (Equal-Weighted)": (f_silo_eq_probs >= 0.5).astype(int),
            "Federated Silos (FedProx)": (f_prox_silo_probs >= 0.5).astype(int),
            "Federated Dir(0.1) (FedAvg)": (f_d01_probs >= 0.5).astype(int),
            "Federated Dir(0.1) (FedProx)": (f_prox_d01_probs >= 0.5).astype(int),
        }
        for att_cat in ["DoS", "PortScan", "Web Attack"]:
            idx_cat = test_meta[test_meta["Category"] == att_cat].index
            for model_label, p_arr in preds_eval_dict.items():
                rec = float(p_arr[idx_cat].mean())
                per_attack_runs.append({
                    "Seed": seed,
                    "Attack": att_cat,
                    "Model": model_label,
                    "Recall": rec,
                    "Total_Flows": len(idx_cat)
                })

    # Save Definitive Comparison Table with Mean and Spread
    print("\n[*] Aggregating 5-seed metrics across all setups...", flush=True)
    summary_rows = []
    for setup in setups:
        runs = all_runs[setup]
        p_vals = [r["precision"] for r in runs]
        r_vals = [r["recall"] for r in runs]
        f1_vals = [r["f1"] for r in runs]
        pr_vals = [r["pr_auc"] for r in runs]
        fpr_vals = [r["fpr"] for r in runs]
        
        summary_rows.append({
            "Setup": setup,
            "Precision_Mean": round(float(np.mean(p_vals)), 4),
            "Precision_Std": round(float(np.std(p_vals)), 4),
            "Recall_Mean": round(float(np.mean(r_vals)), 4),
            "Recall_Std": round(float(np.std(r_vals)), 4),
            "F1_Mean": round(float(np.mean(f1_vals)), 4),
            "F1_Std": round(float(np.std(f1_vals)), 4),
            "PR_AUC_Mean": round(float(np.mean(pr_vals)), 4),
            "PR_AUC_Std": round(float(np.std(pr_vals)), 4),
            "FPR_Mean": round(float(np.mean(fpr_vals)), 4),
            "FPR_Std": round(float(np.std(fpr_vals)), 4),
            "Precision": f"{np.mean(p_vals):.4f} +/- {np.std(p_vals):.4f}",
            "Recall": f"{np.mean(r_vals):.4f} +/- {np.std(r_vals):.4f}",
            "F1": f"{np.mean(f1_vals):.4f} +/- {np.std(f1_vals):.4f}",
            "PR-AUC": f"{np.mean(pr_vals):.4f} +/- {np.std(pr_vals):.4f}",
            "False positive rate": f"{np.mean(fpr_vals):.4f} +/- {np.std(fpr_vals):.4f}"
        })
        
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(EXPERIMENTS_DIR / "rigorous_comparison_table.csv", index=False)
    
    # Save Convergence & Per-Attack Data
    pd.DataFrame(all_convergence_records).to_csv(EXPERIMENTS_DIR / "convergence_history.csv", index=False)
    
    # Aggregate Per-Attack Recall (Mean +/- Std across 5 seeds)
    df_att = pd.DataFrame(per_attack_runs)
    att_summary = []
    for (att, mod), grp in df_att.groupby(["Attack", "Model"]):
        mean_rec = float(grp["Recall"].mean())
        std_rec = float(grp["Recall"].std())
        att_summary.append({
            "Attack": att,
            "Model": mod,
            "Recall_Mean": round(mean_rec, 4),
            "Recall_Std": round(std_rec, 4),
            "Recall": round(mean_rec, 4),
            "Recall_Display": f"{mean_rec:.4f} +/- {std_rec:.4f}",
            "Total_Flows": int(grp["Total_Flows"].iloc[0])
        })
    df_att_summary = pd.DataFrame(att_summary)
    df_att_summary.to_csv(EXPERIMENTS_DIR / "per_attack_recall.csv", index=False)
    
    # Save Communication Overhead
    comm_df = pd.DataFrame([
        {
            "Round": r,
            "Per_Client_Uplink_KB": round(param_bytes / 1024, 2),
            "Total_Network_Traffic_KB": round(3 * 2 * (param_bytes / 1024) * r, 2)
        }
        for r in range(1, num_rounds + 1)
    ])
    comm_df.to_csv(EXPERIMENTS_DIR / "communication_overhead.csv", index=False)
    
    # Save the deployment model for Streamlit
    if last_global_model is not None:
        torch.save(last_global_model.state_dict(), EXPERIMENTS_DIR / "global_federated_model.pth")
        
    print("\n" + "="*115, flush=True)
    print("        DEFINITIVE TIGHTENED MONTH 5 COMPARISON TABLE (5 SEEDS, ZERO TEST LEAKAGE)        ", flush=True)
    print("="*115, flush=True)
    print(summary_df[["Setup", "Precision", "Recall", "F1", "PR-AUC", "False positive rate"]].to_string(index=False), flush=True)
    print("="*115 + "\n", flush=True)

if __name__ == "__main__":
    main()
