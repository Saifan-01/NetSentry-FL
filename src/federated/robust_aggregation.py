"""
NetSentry-FL: Byzantine & Poisoning Robustness Benchmark
Evaluates multi-client defense against a Label-Flipping Poisoning Attack:
- Threat Model: 1 Malicious Client flips binary labels (0 <-> 1)
- Aggregation Strategies:
  1. Standard FedAvg (Mean) - Vulnerable to malicious gradient poisoning
  2. Coordinate-wise Median - Robust to extreme coordinate deviations
  3. Trimmed Mean (trims extreme upper and lower parameter percentiles)
- Network Configurations:
  - 3-Client Setup (1 Malicious = 33.3% of network; near theoretical breakdown limit)
  - 5-Client Setup (1 Malicious = 20.0% of network; below breakdown limit)

Evaluated across the 7,292 held-out test flows across all 5 seeds.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, precision_recall_curve, auc

from src.models.ids_net import NetSentryClassifier, get_model_parameters, set_model_parameters
from src.federated.client import NetSentryClient
from src.models.evaluate_month5_tightened import evaluate_metrics

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHARTS_DIR = PROJECT_ROOT / "reports" / "charts"

def aggregate_fedavg(updates):
    total = sum(n for _, n, _ in updates)
    weights = [n / total for _, n, _ in updates]
    num_layers = len(updates[0][0])
    avg_params = []
    for l in range(num_layers):
        layer_avg = sum(w * up[l] for (up, _, _), w in zip(updates, weights))
        avg_params.append(layer_avg)
    return avg_params

def aggregate_coordinate_median(updates):
    num_layers = len(updates[0][0])
    med_params = []
    for l in range(num_layers):
        stacked = np.stack([up[l] for up, _, _ in updates], axis=0)
        med_params.append(np.median(stacked, axis=0))
    return med_params

def aggregate_trimmed_mean(updates, trim_ratio=0.2):
    num_layers = len(updates[0][0])
    trimmed_params = []
    K = len(updates)
    k_trim = max(1, int(K * trim_ratio))
    for l in range(num_layers):
        stacked = np.stack([up[l] for up, _, _ in updates], axis=0)
        sorted_arr = np.sort(stacked, axis=0)
        if K > 2 * k_trim:
            trimmed = sorted_arr[k_trim : K - k_trim]
        else:
            trimmed = sorted_arr
        trimmed_params.append(np.mean(trimmed, axis=0))
    return trimmed_params

def run_poisoning_simulation(clients_dict, malicious_cids, input_dim, X_test, y_test,
                             aggregator_type="fedavg", num_rounds=15, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    clients = []
    for cid in range(len(clients_dict)):
        X_c = clients_dict[cid]["X"]
        y_c = clients_dict[cid]["y"].copy()
        if cid in malicious_cids:
            # Poisoning: Flip labels 0 -> 1 and 1 -> 0
            y_c = 1.0 - y_c
        clients.append(NetSentryClient(cid, X_c, y_c, input_dim))
        
    global_model = NetSentryClassifier(input_dim=input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.1)
    global_params = get_model_parameters(global_model)
    config = {"local_epochs": 2, "batch_size": 128, "lr": 0.001, "proximal_mu": 0.0}
    
    for rnd in range(1, num_rounds + 1):
        updates = []
        for client in clients:
            up, num, _ = client.fit(global_params, config)
            updates.append((up, num, {}))
            
        if aggregator_type == "fedavg":
            global_params = aggregate_fedavg(updates)
        elif aggregator_type == "median":
            global_params = aggregate_coordinate_median(updates)
        elif aggregator_type == "trimmed_mean":
            global_params = aggregate_trimmed_mean(updates, trim_ratio=0.2)
        set_model_parameters(global_model, global_params)
        
    global_model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(global_model(torch.from_numpy(X_test))).numpy()
    return evaluate_metrics(y_test, probs)

def main():
    torch.set_num_threads(min(12, os.cpu_count() or 4))
    print("="*95, flush=True)
    print("      NETSENTRY-FL: BYZANTINE & LABEL-FLIPPING ROBUSTNESS BENCHMARK (5 SEEDS)      ", flush=True)
    print("="*95, flush=True)

    X_train = np.load(PROCESSED_DIR / "X_train.npy").astype(np.float32)
    y_train = np.load(PROCESSED_DIR / "y_train.npy").astype(np.float32)
    X_test = np.load(PROCESSED_DIR / "X_test.npy").astype(np.float32)
    y_test = np.load(PROCESSED_DIR / "y_test.npy").astype(np.float32)
    input_dim = X_train.shape[1]

    seeds = [42, 101, 777, 1234, 5678]

    # Evaluated Setups
    experiments = [
        # 3 Clients Setup (1 Malicious = 33.3%)
        {"name": "No Attack (3 Clients, FedAvg)", "num_clients": 3, "malicious": [], "agg": "fedavg"},
        {"name": "Label-Flip Attack (3 Clients, FedAvg)", "num_clients": 3, "malicious": [0], "agg": "fedavg"},
        {"name": "Label-Flip Attack (3 Clients, Coordinate Median)", "num_clients": 3, "malicious": [0], "agg": "median"},
        {"name": "Label-Flip Attack (3 Clients, Trimmed Mean)", "num_clients": 3, "malicious": [0], "agg": "trimmed_mean"},
        # 5 Clients Setup (1 Malicious = 20.0%)
        {"name": "No Attack (5 Clients, FedAvg)", "num_clients": 5, "malicious": [], "agg": "fedavg"},
        {"name": "Label-Flip Attack (5 Clients, FedAvg)", "num_clients": 5, "malicious": [0], "agg": "fedavg"},
        {"name": "Label-Flip Attack (5 Clients, Coordinate Median)", "num_clients": 5, "malicious": [0], "agg": "median"},
        {"name": "Label-Flip Attack (5 Clients, Trimmed Mean)", "num_clients": 5, "malicious": [0], "agg": "trimmed_mean"}
    ]

    records = []

    for exp in experiments:
        ename = exp["name"]
        print(f"\n>>> Running Setup: {ename} <<<", flush=True)
        runs = []
        for s in seeds:
            idx_perm = np.random.RandomState(s).permutation(len(X_train))
            splits = np.array_split(idx_perm, exp["num_clients"])
            cdict = {cid: {"X": X_train[splits[cid]], "y": y_train[splits[cid]]} for cid in range(exp["num_clients"])}
            
            m = run_poisoning_simulation(
                cdict, exp["malicious"], input_dim, X_test, y_test,
                aggregator_type=exp["agg"], num_rounds=15, seed=s
            )
            runs.append(m)
            print(f"  [Seed {s}] F1: {m['f1']:.4f} | Recall: {m['recall']:.4f} | FPR: {m['fpr']*100:.2f}%", flush=True)

        f1_vals = [r["f1"] for r in runs]
        rec_vals = [r["recall"] for r in runs]
        prec_vals = [r["precision"] for r in runs]
        fpr_vals = [r["fpr"] for r in runs]
        
        records.append({
            "Experiment": ename,
            "Total_Clients": exp["num_clients"],
            "Malicious_Clients": len(exp["malicious"]),
            "Malicious_Ratio": f"{len(exp['malicious'])/exp['num_clients']*100:.1f}%",
            "Aggregator": exp["agg"],
            "F1_Mean": round(float(np.mean(f1_vals)), 4),
            "F1_Std": round(float(np.std(f1_vals)), 4),
            "Recall_Mean": round(float(np.mean(rec_vals)), 4),
            "Recall_Std": round(float(np.std(rec_vals)), 4),
            "Precision_Mean": round(float(np.mean(prec_vals)), 4),
            "Precision_Std": round(float(np.std(prec_vals)), 4),
            "FPR_Mean": round(float(np.mean(fpr_vals)), 4),
            "FPR_Std": round(float(np.std(fpr_vals)), 4),
            "F1": f"{np.mean(f1_vals):.4f} +/- {np.std(f1_vals):.4f}",
            "Recall": f"{np.mean(rec_vals):.4f} +/- {np.std(rec_vals):.4f}",
            "False_Alarm_Rate": f"{np.mean(fpr_vals):.4f} +/- {np.std(fpr_vals):.4f}"
        })

    df = pd.DataFrame(records)
    df.to_csv(EXPERIMENTS_DIR / "robust_aggregation_benchmark.csv", index=False)
    
    # Plot Poisoning Defense Chart
    import matplotlib.pyplot as plt
    plt.style.use("tableau-colorblind10")
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})
    
    plt.figure(figsize=(9.5, 4.8))
    x = np.arange(len(df))
    plt.bar(x, df["F1_Mean"], yerr=df["F1_Std"], capsize=3, color=["#2ca02c", "#d62728", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#1f77b4", "#ff7f0e"])
    plt.xticks(x, df["Experiment"], rotation=25, ha="right", fontsize=8.5)
    plt.ylabel("Test F1-Score", fontsize=11)
    plt.title("Byzantine Robustness against 1 Label-Flipping Client (3 vs. 5 Clients, Mean +/- Std)", fontsize=11, weight="bold", pad=10)
    plt.ylim(0, 1.1)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    chart_p = CHARTS_DIR / "poisoning_robustness.png"
    plt.savefig(chart_p, dpi=300)
    plt.close()
    
    print("\n" + "="*115, flush=True)
    print("                     POISONING ROBUSTNESS BENCHMARK (5 SEEDS)                     ", flush=True)
    print("="*115, flush=True)
    print(df[["Experiment", "Malicious_Ratio", "Aggregator", "F1", "Recall", "False_Alarm_Rate"]].to_string(index=False), flush=True)
    print("="*115 + "\n", flush=True)

if __name__ == "__main__":
    main()
