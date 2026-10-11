"""
NetSentry-FL: Realistic Non-IID Attack-Skewed Evaluation Harness
Under extreme Non-IID, Client A sees PortScan attacks, Client B sees DoS attacks,
and Client C sees DDoS attacks. When evaluated on the general held-out test set
containing ALL attack types:
- Local models suffer lower recall on attacks they NEVER saw in training.
- Federated FedAvg models collaborate without sharing data and learn to detect ALL attacks!
- Non-IID slows convergence across communication rounds compared to IID.
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

def run_realistic_skewed_evaluation():
    seeds = [42, 101, 777]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    raw_df = pd.read_csv("data/raw/CIC_IDS2017_benchmark_sample.csv")
    scaler = joblib.load(PROCESSED_DIR / "scaler.pkl")
    feature_names = joblib.load(PROCESSED_DIR / "feature_names.pkl")
    
    # Standard split
    n_total = len(raw_df)
    n_train = 28000
    n_test = 8000
    
    train_df = raw_df.iloc[:n_train].copy().reset_index(drop=True)
    test_df = raw_df.iloc[-n_test:].copy().reset_index(drop=True)
    
    X_test = scaler.transform(test_df[feature_names].values).astype(np.float32)
    y_test = (test_df["Label"].str.upper() != "BENIGN").astype(int).values.astype(np.float32)
    input_dim = len(feature_names)
    
    dummy_model = NetSentryClassifier(input_dim=input_dim)
    param_bytes = sum(p.numel() * 4 for p in dummy_model.parameters())
    
    setups = ["Centralized (MLP)", "Local-only (Client A)", "Local-only (Client B)", "Local-only (Client C)", "Federated, IID", "Federated, non-IID"]
    all_runs = {s: [] for s in setups}
    convergence_records = []
    per_attack_records = []

    for seed in seeds:
        print(f"\n==================== RUNNING SEED {seed} ====================")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # 1. Centralized MLP (sees all training data)
        X_train = scaler.transform(train_df[feature_names].values).astype(np.float32)
        y_train = (train_df["Label"].str.upper() != "BENIGN").astype(int).values.astype(np.float32)
        
        cent_model = NetSentryClassifier(input_dim=input_dim)
        opt = torch.optim.AdamW(cent_model.parameters(), lr=0.001, weight_decay=1e-4)
        crit = nn.BCEWithLogitsLoss()
        loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=128, shuffle=True)
        
        for _ in range(8):
            for bx, by in loader:
                opt.zero_grad()
                crit(cent_model(bx), by).backward()
                opt.step()
                
        cent_model.eval()
        with torch.no_grad():
            cent_probs = torch.sigmoid(cent_model(torch.from_numpy(X_test))).numpy()
        all_runs["Centralized (MLP)"].append(evaluate_metrics(y_test, cent_probs))
        
        # 2. Extreme Non-IID Partitions
        # Client 0: Benign + ONLY PortScan
        # Client 1: Benign + ONLY DoS Hulk
        # Client 2: Benign + ONLY DDoS & FTP-Patator
        client_parts = {0: {"X": [], "y": []}, 1: {"X": [], "y": []}, 2: {"X": [], "y": []}}
        
        for idx_row, row in train_df.iterrows():
            lbl = str(row["Label"]).strip()
            feat = row[feature_names].values.astype(np.float32)
            
            if lbl == "BENIGN":
                c = np.random.choice([0, 1, 2])
                client_parts[c]["X"].append(feat)
                client_parts[c]["y"].append(0.0)
            elif lbl == "PortScan":
                # Client 0 gets 95% of PortScans
                c = np.random.choice([0, 1, 2], p=[0.95, 0.03, 0.02])
                client_parts[c]["X"].append(feat)
                client_parts[c]["y"].append(1.0)
            elif lbl == "DoS Hulk":
                # Client 1 gets 95% of DoS Hulk
                c = np.random.choice([0, 1, 2], p=[0.02, 0.95, 0.03])
                client_parts[c]["X"].append(feat)
                client_parts[c]["y"].append(1.0)
            else: # DDoS and FTP-Patator
                # Client 2 gets 95% of DDoS/FTP
                c = np.random.choice([0, 1, 2], p=[0.02, 0.03, 0.95])
                client_parts[c]["X"].append(feat)
                client_parts[c]["y"].append(1.0)
                
        for c in range(3):
            client_parts[c]["X"] = scaler.transform(np.array(client_parts[c]["X"])).astype(np.float32)
            client_parts[c]["y"] = np.array(client_parts[c]["y"], dtype=np.float32)

        # Local-Only Models Evaluation on Held-out test set
        local_preds_dict = {}
        for cid, cname in enumerate(["Local-only (Client A)", "Local-only (Client B)", "Local-only (Client C)"]):
            l_model = NetSentryClassifier(input_dim=input_dim)
            l_opt = torch.optim.AdamW(l_model.parameters(), lr=0.001, weight_decay=1e-4)
            l_loader = DataLoader(
                TensorDataset(torch.from_numpy(client_parts[cid]["X"]), torch.from_numpy(client_parts[cid]["y"])),
                batch_size=64, shuffle=True
            )
            for _ in range(5):
                for bx, by in l_loader:
                    l_opt.zero_grad()
                    crit(l_model(bx), by).backward()
                    l_opt.step()
            l_model.eval()
            with torch.no_grad():
                l_probs = torch.sigmoid(l_model(torch.from_numpy(X_test))).numpy()
            local_preds_dict[cid] = l_probs
            all_runs[cname].append(evaluate_metrics(y_test, l_probs))

        # 3. Federated IID (Uniform split)
        idx_perm = np.random.permutation(len(X_train))
        iid_splits = np.array_split(idx_perm, 3)
        fed_iid_model = NetSentryClassifier(input_dim=input_dim)
        fed_iid_params = get_model_parameters(fed_iid_model)
        
        for rnd in range(1, 11):
            updates = []
            for cid in range(3):
                client = NetSentryClient(cid, X_train[iid_splits[cid]], y_train[iid_splits[cid]], input_dim)
                up, num, _ = client.fit(fed_iid_params, {"local_epochs": 2, "batch_size": 64, "lr": 0.002})
                updates.append((up, num, {}))
            fed_iid_params = aggregate_fedavg(updates)
            set_model_parameters(fed_iid_model, fed_iid_params)
            
            fed_iid_model.eval()
            with torch.no_grad():
                f_probs = torch.sigmoid(fed_iid_model(torch.from_numpy(X_test))).numpy()
            m_round = evaluate_metrics(y_test, f_probs)
            convergence_records.append({"seed": seed, "round": rnd, "setup": "Federated, IID", "f1": m_round["f1"], "recall": m_round["recall"]})
            
        all_runs["Federated, IID"].append(evaluate_metrics(y_test, f_probs))

        # 4. Federated Non-IID (Collaborative learning across heterogeneous silos)
        fed_non_iid_model = NetSentryClassifier(input_dim=input_dim)
        fed_non_iid_params = get_model_parameters(fed_non_iid_model)
        for rnd in range(1, 11):
            updates = []
            for cid in range(3):
                client = NetSentryClient(cid, client_parts[cid]["X"], client_parts[cid]["y"], input_dim)
                up, num, _ = client.fit(fed_non_iid_params, {"local_epochs": 2, "batch_size": 64, "lr": 0.002})
                updates.append((up, num, {}))
            fed_non_iid_params = aggregate_fedavg(updates)
            set_model_parameters(fed_non_iid_model, fed_non_iid_params)
            
            fed_non_iid_model.eval()
            with torch.no_grad():
                fn_probs = torch.sigmoid(fed_non_iid_model(torch.from_numpy(X_test))).numpy()
            m_round = evaluate_metrics(y_test, fn_probs)
            convergence_records.append({"seed": seed, "round": rnd, "setup": "Federated, non-IID", "f1": m_round["f1"], "recall": m_round["recall"]})

        all_runs["Federated, non-IID"].append(evaluate_metrics(y_test, fn_probs))

        # Per-attack breakdown recorded for Seed 42
        if seed == 42:
            preds_dict = {
                "Centralized": (cent_probs >= 0.5).astype(int),
                "Local Client A (PortScan)": (local_preds_dict[0] >= 0.5).astype(int),
                "Local Client B (DoS)": (local_preds_dict[1] >= 0.5).astype(int),
                "Local Client C (DDoS)": (local_preds_dict[2] >= 0.5).astype(int),
                "Federated IID": (f_probs >= 0.5).astype(int),
                "Federated Non-IID": (fn_probs >= 0.5).astype(int),
            }
            for att_type in ["PortScan", "DoS Hulk", "DDoS", "FTP-Patator"]:
                idx_att = test_df[test_df["Label"] == att_type].index
                if len(idx_att) > 0:
                    for model_name, p_vec in preds_dict.items():
                        rec = p_vec[idx_att].mean()
                        per_attack_records.append({
                            "Attack": att_type,
                            "Model": model_name,
                            "Recall": rec,
                            "Total_Flows": len(idx_att)
                        })

    # Save summary table (averages over 3 seeds)
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
    
    pd.DataFrame(convergence_records).to_csv(EXPERIMENTS_DIR / "convergence_history.csv", index=False)
    pd.DataFrame(per_attack_records).to_csv(EXPERIMENTS_DIR / "per_attack_recall.csv", index=False)
    
    # Save the final deployed models for Streamlit app
    torch.save(fed_non_iid_model.state_dict(), EXPERIMENTS_DIR / "global_federated_model.pth")
    torch.save(cent_model.state_dict(), EXPERIMENTS_DIR / "centralized_model.pth")
    
    print("\n" + "="*80)
    print("                    FINAL 3-SEED RIGOROUS COMPARISON TABLE")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80 + "\n")

if __name__ == "__main__":
    run_realistic_skewed_evaluation()
