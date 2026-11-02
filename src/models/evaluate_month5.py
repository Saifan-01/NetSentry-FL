"""
NetSentry-FL: Month 5 Core Federated Heterogeneity Evaluation Harness
Evaluates:
1. Centralized MLP Baseline (with Gaussian Quantile Scaling)
2. Local-only Baselines (Clients A, B, C under Extreme Attack Silos)
3. Federated IID (FedAvg, 15 rounds)
4. Federated Non-IID Extreme Silos (FedAvg vs. FedProx mu=0.01 vs. FedProx mu=0.1)
5. Federated Non-IID Dirichlet Label Skew (alpha = 0.5, 0.1; FedAvg vs. FedProx mu=0.01 vs. FedProx mu=0.1)

Averaged across 3 random seeds (42, 101, 777).
Outputs:
- Mean +/- std for Precision, Recall, F1, PR-AUC, and False Alarm Rate (FPR).
- Per-attack-type recall (DoS, PortScan, Web Attack) with mean +/- std across 3 seeds.
- Round-by-round convergence histories for all federated strategies.
- Communication overhead (KB transferred across rounds).
- Saves global model artifact for Streamlit deployment.
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
from src.federated.run_federated_experiment import aggregate_fedavg
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

def run_fl_simulation(
    clients_dict: dict,
    input_dim: int,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_rounds: int = 15,
    local_epochs: int = 2,
    lr: float = 0.001,
    proximal_mu: float = 0.0,
    setup_name: str = "FL",
    seed: int = 42
):
    """Orchestrates multi-round federated training with FedAvg or FedProx."""
    clients = [
        NetSentryClient(cid, clients_dict[cid]["X"], clients_dict[cid]["y"], input_dim)
        for cid in range(len(clients_dict))
    ]
    global_model = NetSentryClassifier(input_dim=input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.1)
    global_params = get_model_parameters(global_model)
    
    round_history = []
    config = {
        "local_epochs": local_epochs,
        "batch_size": 64,
        "lr": lr,
        "proximal_mu": proximal_mu
    }
    
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
        m = evaluate_metrics(y_test, probs)
        round_history.append({
            "seed": seed,
            "round": rnd,
            "setup": setup_name,
            "f1": m["f1"],
            "recall": m["recall"],
            "precision": m["precision"]
        })
        
    return global_model, probs, round_history

def run_month5_full_evaluation():
    seeds = [42, 101, 777]
    num_rounds = 15
    local_epochs = 2
    
    print("[*] Loading 70/10/20 real CIC-IDS2017 curated splits...", flush=True)
    X_train = np.load(PROCESSED_DIR / "X_train.npy").astype(np.float32)
    y_train = np.load(PROCESSED_DIR / "y_train.npy").astype(np.float32)
    X_test = np.load(PROCESSED_DIR / "X_test.npy").astype(np.float32)
    y_test = np.load(PROCESSED_DIR / "y_test.npy").astype(np.float32)
    
    train_meta = pd.read_csv(PROCESSED_DIR / "train_metadata.csv")
    test_meta = pd.read_csv(PROCESSED_DIR / "test_metadata.csv")
    input_dim = X_train.shape[1]
    
    # 1. Generate client class distribution plots
    print("[*] Generating publication partition distribution plots...", flush=True)
    silo_extreme_demo = create_extreme_attack_silos(X_train, train_meta, seed=42)
    silo_dir_05_demo = create_dirichlet_partitions(X_train, train_meta, alpha=0.5, seed=42)
    silo_dir_01_demo = create_dirichlet_partitions(X_train, train_meta, alpha=0.1, seed=42)
    plot_client_class_distributions({
        "Extreme Attack Silos": silo_extreme_demo,
        "Dirichlet (alpha = 0.5)": silo_dir_05_demo,
        "Dirichlet (alpha = 0.1)": silo_dir_01_demo
    }, CHARTS_DIR / "dirichlet_and_silo_distributions.png")
    
    dummy_model = NetSentryClassifier(input_dim=input_dim)
    param_bytes = sum(p.numel() * 4 for p in dummy_model.parameters())
    print(f"[*] NetSentryClassifier: {input_dim} features | Parameters: {sum(p.numel() for p in dummy_model.parameters()):,} ({param_bytes / 1024:.2f} KB)", flush=True)

    setups = [
        "Centralized (MLP)",
        "Local-only (Client A - DoS)",
        "Local-only (Client B - PortScan)",
        "Local-only (Client C - Web Attack)",
        "Federated, IID (FedAvg)",
        "Federated, non-IID Silos (FedAvg)",
        "Federated, non-IID Silos (FedProx mu=0.01)",
        "Federated, non-IID Silos (FedProx mu=0.1)",
        "Federated, non-IID Dir(0.5) (FedAvg)",
        "Federated, non-IID Dir(0.1) (FedAvg)",
        "Federated, non-IID Dir(0.1) (FedProx mu=0.01)",
        "Federated, non-IID Dir(0.1) (FedProx mu=0.1)"
    ]
    all_runs = {s: [] for s in setups}
    all_convergence_records = []
    per_attack_runs = []

    last_fed_iid_model = None

    for seed in seeds:
        print(f"\n==================== RUNNING MONTH 5 EXPERIMENTS (SEED {seed}) ====================", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # 1. Centralized MLP Baseline
        print(f"[{seed}] Training Centralized MLP Baseline...", flush=True)
        cent_model = NetSentryClassifier(input_dim=input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.1)
        opt_cent = torch.optim.AdamW(cent_model.parameters(), lr=0.001, weight_decay=1e-4)
        pos_wt_val = min((len(y_train) - y_train.sum()) / (y_train.sum() + 1e-5), 10.0)
        crit_cent = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_wt_val]))
        loader_cent = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=128, shuffle=True)
        
        cent_model.train()
        for epoch in range(15):
            for bx, by in loader_cent:
                opt_cent.zero_grad()
                crit_cent(cent_model(bx), by).backward()
                opt_cent.step()
                
        cent_model.eval()
        with torch.no_grad():
            cent_probs = torch.sigmoid(cent_model(torch.from_numpy(X_test))).numpy()
        cent_m = evaluate_metrics(y_test, cent_probs)
        all_runs["Centralized (MLP)"].append(cent_m)
        print(f"     -> Centralized MLP: F1={cent_m['f1']:.4f}, Recall={cent_m['recall']:.4f}, FPR={cent_m['fpr']*100:.2f}%", flush=True)

        # 2. Local-Only Models (Extreme Silos)
        print(f"[{seed}] Training Local-Only Silo Models (Clients A, B, C)...", flush=True)
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
                batch_size=64, shuffle=True
            )
            l_model.train()
            for _ in range(8):
                for bx, by in l_loader:
                    l_opt.zero_grad()
                    l_crit(l_model(bx), by).backward()
                    l_opt.step()
                    
            l_model.eval()
            with torch.no_grad():
                l_probs = torch.sigmoid(l_model(torch.from_numpy(X_test))).numpy()
            local_probs_dict[cid] = l_probs
            lm = evaluate_metrics(y_test, l_probs)
            all_runs[sname].append(lm)
            print(f"     -> {sname}: F1={lm['f1']:.4f}, Recall={lm['recall']:.4f}, FPR={lm['fpr']*100:.2f}%", flush=True)

        # 3. Federated IID (FedAvg, 15 rounds)
        print(f"[{seed}] Training Federated IID (FedAvg)...", flush=True)
        idx_perm = np.random.permutation(len(X_train))
        iid_splits = np.array_split(idx_perm, 3)
        iid_clients_dict = {
            cid: {"X": X_train[iid_splits[cid]], "y": y_train[iid_splits[cid]]}
            for cid in range(3)
        }
        fed_iid_model, f_iid_probs, hist_iid = run_fl_simulation(
            iid_clients_dict, input_dim, X_test, y_test, num_rounds=num_rounds,
            local_epochs=local_epochs, lr=0.001, proximal_mu=0.0,
            setup_name="Federated, IID (FedAvg)", seed=seed
        )
        last_fed_iid_model = fed_iid_model
        all_runs["Federated, IID (FedAvg)"].append(evaluate_metrics(y_test, f_iid_probs))
        all_convergence_records.extend(hist_iid)

        # 4. Federated Non-IID Extreme Silos (FedAvg)
        print(f"[{seed}] Training Federated Non-IID Silos (FedAvg)...", flush=True)
        fed_silo_model, f_silo_probs, hist_silo = run_fl_simulation(
            extreme_silos, input_dim, X_test, y_test, num_rounds=num_rounds,
            local_epochs=local_epochs, lr=0.001, proximal_mu=0.0,
            setup_name="Federated, non-IID Silos (FedAvg)", seed=seed
        )
        all_runs["Federated, non-IID Silos (FedAvg)"].append(evaluate_metrics(y_test, f_silo_probs))
        all_convergence_records.extend(hist_silo)

        # 5. Federated Non-IID Extreme Silos with FedProx (mu = 0.01)
        print(f"[{seed}] Training Federated Non-IID Silos (FedProx mu=0.01)...", flush=True)
        fed_prox_silo_01_model, f_prox_silo_01_probs, hist_prox_silo_01 = run_fl_simulation(
            extreme_silos, input_dim, X_test, y_test, num_rounds=num_rounds,
            local_epochs=local_epochs, lr=0.001, proximal_mu=0.01,
            setup_name="Federated, non-IID Silos (FedProx mu=0.01)", seed=seed
        )
        all_runs["Federated, non-IID Silos (FedProx mu=0.01)"].append(evaluate_metrics(y_test, f_prox_silo_01_probs))
        all_convergence_records.extend(hist_prox_silo_01)

        # 6. Federated Non-IID Extreme Silos with FedProx (mu = 0.1)
        print(f"[{seed}] Training Federated Non-IID Silos (FedProx mu=0.1)...", flush=True)
        fed_prox_silo_10_model, f_prox_silo_10_probs, hist_prox_silo_10 = run_fl_simulation(
            extreme_silos, input_dim, X_test, y_test, num_rounds=num_rounds,
            local_epochs=local_epochs, lr=0.001, proximal_mu=0.1,
            setup_name="Federated, non-IID Silos (FedProx mu=0.1)", seed=seed
        )
        all_runs["Federated, non-IID Silos (FedProx mu=0.1)"].append(evaluate_metrics(y_test, f_prox_silo_10_probs))
        all_convergence_records.extend(hist_prox_silo_10)

        # 7. Federated Non-IID Dirichlet (alpha = 0.5) (FedAvg)
        print(f"[{seed}] Training Federated Non-IID Dir(0.5) (FedAvg)...", flush=True)
        dir_05_silos = create_dirichlet_partitions(X_train, train_meta, alpha=0.5, seed=seed)
        fed_d05_model, f_d05_probs, hist_d05 = run_fl_simulation(
            dir_05_silos, input_dim, X_test, y_test, num_rounds=num_rounds,
            local_epochs=local_epochs, lr=0.001, proximal_mu=0.0,
            setup_name="Federated, non-IID Dir(0.5) (FedAvg)", seed=seed
        )
        all_runs["Federated, non-IID Dir(0.5) (FedAvg)"].append(evaluate_metrics(y_test, f_d05_probs))
        all_convergence_records.extend(hist_d05)

        # 8. Federated Non-IID Dirichlet (alpha = 0.1) (FedAvg)
        print(f"[{seed}] Training Federated Non-IID Dir(0.1) (FedAvg)...", flush=True)
        dir_01_silos = create_dirichlet_partitions(X_train, train_meta, alpha=0.1, seed=seed)
        fed_d01_model, f_d01_probs, hist_d01 = run_fl_simulation(
            dir_01_silos, input_dim, X_test, y_test, num_rounds=num_rounds,
            local_epochs=local_epochs, lr=0.001, proximal_mu=0.0,
            setup_name="Federated, non-IID Dir(0.1) (FedAvg)", seed=seed
        )
        all_runs["Federated, non-IID Dir(0.1) (FedAvg)"].append(evaluate_metrics(y_test, f_d01_probs))
        all_convergence_records.extend(hist_d01)

        # 9. Federated Non-IID Dirichlet (alpha = 0.1) with FedProx (mu = 0.01)
        print(f"[{seed}] Training Federated Non-IID Dir(0.1) (FedProx mu=0.01)...", flush=True)
        fed_prox_d01_01_model, f_prox_d01_01_probs, hist_prox_d01_01 = run_fl_simulation(
            dir_01_silos, input_dim, X_test, y_test, num_rounds=num_rounds,
            local_epochs=local_epochs, lr=0.001, proximal_mu=0.01,
            setup_name="Federated, non-IID Dir(0.1) (FedProx mu=0.01)", seed=seed
        )
        all_runs["Federated, non-IID Dir(0.1) (FedProx mu=0.01)"].append(evaluate_metrics(y_test, f_prox_d01_01_probs))
        all_convergence_records.extend(hist_prox_d01_01)

        # 10. Federated Non-IID Dirichlet (alpha = 0.1) with FedProx (mu = 0.1)
        print(f"[{seed}] Training Federated Non-IID Dir(0.1) (FedProx mu=0.1)...", flush=True)
        fed_prox_d01_10_model, f_prox_d01_10_probs, hist_prox_d01_10 = run_fl_simulation(
            dir_01_silos, input_dim, X_test, y_test, num_rounds=num_rounds,
            local_epochs=local_epochs, lr=0.001, proximal_mu=0.1,
            setup_name="Federated, non-IID Dir(0.1) (FedProx mu=0.1)", seed=seed
        )
        all_runs["Federated, non-IID Dir(0.1) (FedProx mu=0.1)"].append(evaluate_metrics(y_test, f_prox_d01_10_probs))
        all_convergence_records.extend(hist_prox_d01_10)

        # Record Per-Attack-Type Recall on this seed
        preds_eval_dict = {
            "Centralized (MLP)": (cent_probs >= 0.5).astype(int),
            "Local Client A (DoS)": (local_probs_dict[0] >= 0.5).astype(int),
            "Local Client B (PortScan)": (local_probs_dict[1] >= 0.5).astype(int),
            "Local Client C (Web Attack)": (local_probs_dict[2] >= 0.5).astype(int),
            "Federated IID (FedAvg)": (f_iid_probs >= 0.5).astype(int),
            "Federated Silos (FedAvg)": (f_silo_probs >= 0.5).astype(int),
            "Federated Silos (FedProx mu=0.01)": (f_prox_silo_01_probs >= 0.5).astype(int),
            "Federated Silos (FedProx mu=0.1)": (f_prox_silo_10_probs >= 0.5).astype(int),
            "Federated Dir(0.1) (FedAvg)": (f_d01_probs >= 0.5).astype(int),
            "Federated Dir(0.1) (FedProx mu=0.01)": (f_prox_d01_01_probs >= 0.5).astype(int),
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
    print("\n[*] Aggregating 3-seed evaluation metrics across all setups...", flush=True)
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
    
    # Aggregate Per-Attack Recall (Mean +/- Std across 3 seeds)
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
    
    # Save Communication Overhead across rounds
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
    if last_fed_iid_model is not None:
        torch.save(last_fed_iid_model.state_dict(), EXPERIMENTS_DIR / "global_federated_model.pth")
    
    print("\n" + "="*115, flush=True)
    print("               DEFINITIVE MONTH 5 COMPARISON TABLE (MEAN +/- STD, 3 SEEDS)               ", flush=True)
    print("="*115, flush=True)
    print(summary_df[["Setup", "Precision", "Recall", "F1", "PR-AUC", "False positive rate"]].to_string(index=False), flush=True)
    print("="*115 + "\n", flush=True)

if __name__ == "__main__":
    run_month5_full_evaluation()
