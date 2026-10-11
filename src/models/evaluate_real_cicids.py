"""
NetSentry-FL: Rigorous Real CIC-IDS2017 3-Seed Multi-Attack Evaluation Harness
Uses:
- Real CIC-IDS2017 curated traffic (Deduplicated, zero-leakage RobustScaler fitted strictly on train).
- Extreme Non-IID Partition by attack category:
  * Client A (Organization A): Sees ONLY real DoS attacks + Benign
  * Client B (Organization B): Sees ONLY real PortScan attacks + Benign
  * Client C (Organization C): Sees ONLY real Web Attacks + Benign
- Runs 15 Federated Communication Rounds with Flower FedAvg.
- Evaluates across 3 random seeds (42, 101, 777).
- Computes Precision, Recall, F1, PR-AUC, False Positive Rate (FPR),
  Per-Attack Recall across all attack categories, and Communication Overhead.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix
import joblib

from src.models.ids_net import NetSentryClassifier, get_model_parameters, set_model_parameters
from src.federated.client import NetSentryClient
from src.federated.run_federated_experiment import aggregate_fedavg

EXPERIMENTS_DIR = Path("experiments")
PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def create_extreme_non_iid_silos(X_train, train_meta, seed: int = 42):
    """
    Client A: Sees ONLY DoS + Benign
    Client B: Sees ONLY PortScan + Benign
    Client C: Sees ONLY Web Attack + Benign
    Benign flows are evenly distributed across the 3 clients.
    """
    np.random.seed(seed)
    categories = train_meta["Category"].values
    
    benign_indices = np.where(categories == "BENIGN")[0]
    dos_indices = np.where(categories == "DoS")[0]
    ps_indices = np.where(categories == "PortScan")[0]
    web_indices = np.where(categories == "Web Attack")[0]
    
    np.random.shuffle(benign_indices)
    b_splits = np.array_split(benign_indices, 3)
    
    client_indices = {
        0: np.concatenate([b_splits[0], dos_indices]),
        1: np.concatenate([b_splits[1], ps_indices]),
        2: np.concatenate([b_splits[2], web_indices])
    }
    
    client_data = {}
    for cid in range(3):
        idx = client_indices[cid]
        np.random.shuffle(idx)
        client_data[cid] = {
            "X": X_train[idx],
            "y": train_meta["Binary_Label"].values[idx].astype(np.float32)
        }
        
    return client_data

def evaluate_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    p_curve, r_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(r_curve, p_curve)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "fpr": fpr
    }

def run_real_cicids_evaluation():
    seeds = [42, 101, 777]
    num_rounds = 15
    local_epochs = 2
    
    # Load Real Preprocessed Data
    X_train = np.load(PROCESSED_DIR / "X_train.npy").astype(np.float32)
    y_train = np.load(PROCESSED_DIR / "y_train.npy").astype(np.float32)
    X_test = np.load(PROCESSED_DIR / "X_test.npy").astype(np.float32)
    y_test = np.load(PROCESSED_DIR / "y_test.npy").astype(np.float32)
    
    train_meta = pd.read_csv(PROCESSED_DIR / "train_metadata.csv")
    test_meta = pd.read_csv(PROCESSED_DIR / "test_metadata.csv")
    input_dim = X_train.shape[1]
    
    dummy_model = NetSentryClassifier(input_dim=input_dim)
    param_bytes = sum(p.numel() * 4 for p in dummy_model.parameters())
    print(f"[*] Real Data MLP Features: {input_dim} | Model Size: {param_bytes / 1024:.2f} KB")

    setups = [
        "Centralized (MLP)",
        "Local-only (Client A - DoS)",
        "Local-only (Client B - PortScan)",
        "Local-only (Client C - Web Attack)",
        "Federated, IID",
        "Federated, non-IID"
    ]
    all_runs = {s: [] for s in setups}
    convergence_records = []
    per_attack_records = []

    for seed in seeds:
        print(f"\n==================== RUNNING REAL DATA SEED {seed} ====================")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # 1. Centralized MLP (Upper Bound)
        cent_model = NetSentryClassifier(input_dim=input_dim)
        opt_cent = torch.optim.AdamW(cent_model.parameters(), lr=0.001, weight_decay=1e-4)
        pos_wt = torch.tensor([(len(y_train) - y_train.sum()) / (y_train.sum() + 1e-5)])
        crit_cent = nn.BCEWithLogitsLoss(pos_weight=pos_wt)
        loader_cent = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=128, shuffle=True)
        
        cent_model.train()
        for epoch in range(12):
            for bx, by in loader_cent:
                opt_cent.zero_grad()
                crit_cent(cent_model(bx), by).backward()
                opt_cent.step()
                
        cent_model.eval()
        with torch.no_grad():
            cent_probs = torch.sigmoid(cent_model(torch.from_numpy(X_test))).numpy()
        all_runs["Centralized (MLP)"].append(evaluate_metrics(y_test, cent_probs))

        # 2. Extreme Non-IID Silos
        silos = create_extreme_non_iid_silos(X_train, train_meta, seed=seed)
        local_probs_dict = {}
        
        silo_names = [
            "Local-only (Client A - DoS)",
            "Local-only (Client B - PortScan)",
            "Local-only (Client C - Web Attack)"
        ]
        
        for cid, sname in enumerate(silo_names):
            l_model = NetSentryClassifier(input_dim=input_dim)
            l_opt = torch.optim.AdamW(l_model.parameters(), lr=0.001, weight_decay=1e-4)
            l_pos_wt = torch.tensor([(len(silos[cid]["y"]) - silos[cid]["y"].sum()) / (silos[cid]["y"].sum() + 1e-5)])
            l_crit = nn.BCEWithLogitsLoss(pos_weight=l_pos_wt)
            
            l_loader = DataLoader(
                TensorDataset(torch.from_numpy(silos[cid]["X"]), torch.from_numpy(silos[cid]["y"])),
                batch_size=64, shuffle=True
            )
            l_model.train()
            # 8 epochs for local model
            for _ in range(8):
                for bx, by in l_loader:
                    l_opt.zero_grad()
                    l_crit(l_model(bx), by).backward()
                    l_opt.step()
                    
            l_model.eval()
            with torch.no_grad():
                l_probs = torch.sigmoid(l_model(torch.from_numpy(X_test))).numpy()
            local_probs_dict[cid] = l_probs
            all_runs[sname].append(evaluate_metrics(y_test, l_probs))

        # 3. Federated IID (Uniformly distributed data across 3 clients)
        idx_perm = np.random.permutation(len(X_train))
        iid_splits = np.array_split(idx_perm, 3)
        fed_iid_model = NetSentryClassifier(input_dim=input_dim)
        fed_iid_params = get_model_parameters(fed_iid_model)
        
        for rnd in range(1, num_rounds + 1):
            updates = []
            for cid in range(3):
                client = NetSentryClient(cid, X_train[iid_splits[cid]], y_train[iid_splits[cid]], input_dim)
                up, num, _ = client.fit(fed_iid_params, {"local_epochs": local_epochs, "batch_size": 64, "lr": 0.001})
                updates.append((up, num, {}))
            fed_iid_params = aggregate_fedavg(updates)
            set_model_parameters(fed_iid_model, fed_iid_params)
            
            fed_iid_model.eval()
            with torch.no_grad():
                f_probs = torch.sigmoid(fed_iid_model(torch.from_numpy(X_test))).numpy()
            m_rnd = evaluate_metrics(y_test, f_probs)
            convergence_records.append({
                "seed": seed, "round": rnd, "setup": "Federated, IID",
                "f1": m_rnd["f1"], "recall": m_rnd["recall"], "precision": m_rnd["precision"]
            })
        all_runs["Federated, IID"].append(evaluate_metrics(y_test, f_probs))

        # 4. Federated Non-IID (Collaborative training across extreme attack silos)
        fed_non_iid_model = NetSentryClassifier(input_dim=input_dim)
        fed_non_iid_params = get_model_parameters(fed_non_iid_model)
        
        for rnd in range(1, num_rounds + 1):
            updates = []
            for cid in range(3):
                client = NetSentryClient(cid, silos[cid]["X"], silos[cid]["y"], input_dim)
                up, num, _ = client.fit(fed_non_iid_params, {"local_epochs": local_epochs, "batch_size": 64, "lr": 0.001})
                updates.append((up, num, {}))
            fed_non_iid_params = aggregate_fedavg(updates)
            set_model_parameters(fed_non_iid_model, fed_non_iid_params)
            
            fed_non_iid_model.eval()
            with torch.no_grad():
                fn_probs = torch.sigmoid(fed_non_iid_model(torch.from_numpy(X_test))).numpy()
            m_rnd = evaluate_metrics(y_test, fn_probs)
            convergence_records.append({
                "seed": seed, "round": rnd, "setup": "Federated, non-IID",
                "f1": m_rnd["f1"], "recall": m_rnd["recall"], "precision": m_rnd["precision"]
            })
        all_runs["Federated, non-IID"].append(evaluate_metrics(y_test, fn_probs))

        # Per-Attack Recall Evaluation recorded for Seed 42
        if seed == 42:
            preds_dict = {
                "Centralized (MLP)": (cent_probs >= 0.5).astype(int),
                "Local Client A (DoS)": (local_probs_dict[0] >= 0.5).astype(int),
                "Local Client B (PortScan)": (local_probs_dict[1] >= 0.5).astype(int),
                "Local Client C (Web Attack)": (local_probs_dict[2] >= 0.5).astype(int),
                "Federated (IID)": (f_probs >= 0.5).astype(int),
                "Federated (Non-IID)": (fn_probs >= 0.5).astype(int),
            }
            for att_cat in ["DoS", "PortScan", "Web Attack"]:
                idx_cat = test_meta[test_meta["Category"] == att_cat].index
                for model_label, p_arr in preds_dict.items():
                    rec = float(p_arr[idx_cat].mean())
                    per_attack_records.append({
                        "Attack": att_cat,
                        "Model": model_label,
                        "Recall": round(rec, 4),
                        "Total_Flows": len(idx_cat)
                    })

    # Compute Summary Comparison Table with Mean across 3 Seeds
    summary_rows = []
    for setup in setups:
        runs = all_runs[setup]
        summary_rows.append({
            "Setup": setup,
            "Precision": round(float(np.mean([r["precision"] for r in runs])), 4),
            "Recall": round(float(np.mean([r["recall"] for r in runs])), 4),
            "F1": round(float(np.mean([r["f1"] for r in runs])), 4),
            "PR-AUC": round(float(np.mean([r["pr_auc"] for r in runs])), 4),
            "False positive rate": round(float(np.mean([r["fpr"] for r in runs])), 4)
        })
        
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(EXPERIMENTS_DIR / "rigorous_comparison_table.csv", index=False)
    
    # Save Convergence & Per-Attack Data
    pd.DataFrame(convergence_records).to_csv(EXPERIMENTS_DIR / "convergence_history.csv", index=False)
    pd.DataFrame(per_attack_records).to_csv(EXPERIMENTS_DIR / "per_attack_recall.csv", index=False)
    
    # Save Communication Overhead
    comm_df = pd.DataFrame([
        {"Round": r, "Per_Client_Uplink_KB": round(param_bytes / 1024, 2), "Total_Network_Traffic_KB": round(3 * 2 * (param_bytes / 1024) * r, 2)}
        for r in range(1, num_rounds + 1)
    ])
    comm_df.to_csv(EXPERIMENTS_DIR / "communication_overhead.csv", index=False)

    # Save models for Streamlit app
    torch.save(fed_non_iid_model.state_dict(), EXPERIMENTS_DIR / "global_federated_model.pth")
    torch.save(cent_model.state_dict(), EXPERIMENTS_DIR / "centralized_model.pth")

    print("\n" + "="*85)
    print("        DEFINITIVE REAL CIC-IDS2017 3-SEED COMPARISON TABLE (N=9,115 TEST FLOWS)     ")
    print("="*85)
    print(summary_df.to_string(index=False))
    print("="*85 + "\n")

if __name__ == "__main__":
    run_real_cicids_evaluation()
