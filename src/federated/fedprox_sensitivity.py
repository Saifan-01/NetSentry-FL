"""
NetSentry-FL: FedProx Hyperparameter & Local Epoch Sensitivity Exploration
Analyzes the interaction between local epochs (E=2 vs E=5) and proximal regularization (mu = 0.0, 0.001, 0.01, 0.1, 1.0)
under Extreme Attack Silos and Dirichlet alpha=0.1.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from src.models.ids_net import NetSentryClassifier, get_model_parameters, set_model_parameters
from src.federated.client import NetSentryClient
from src.federated.run_federated_experiment import aggregate_fedavg
from src.federated.partitioners import create_extreme_attack_silos, create_dirichlet_partitions

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

def run_fl_trial(clients_dict, input_dim, X_test, y_test, num_rounds=20, local_epochs=2, proximal_mu=0.0, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    clients = [
        NetSentryClient(cid, clients_dict[cid]["X"], clients_dict[cid]["y"], input_dim)
        for cid in range(len(clients_dict))
    ]
    global_model = NetSentryClassifier(input_dim=input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.1)
    global_params = get_model_parameters(global_model)
    
    config = {
        "local_epochs": local_epochs,
        "batch_size": 64,
        "lr": 0.001,
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
    preds = (probs >= 0.5).astype(int)
    
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = f1_score(y_test, preds, zero_division=0)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    
    return {"f1": f1, "precision": prec, "recall": rec, "fpr": fpr}

def main():
    X_train = np.load(PROCESSED_DIR / "X_train.npy").astype(np.float32)
    X_test = np.load(PROCESSED_DIR / "X_test.npy").astype(np.float32)
    y_test = np.load(PROCESSED_DIR / "y_test.npy").astype(np.float32)
    train_meta = pd.read_csv(PROCESSED_DIR / "train_metadata.csv")
    input_dim = X_train.shape[1]
    
    silos = create_extreme_attack_silos(X_train, train_meta, seed=42)
    dir_01 = create_dirichlet_partitions(X_train, train_meta, alpha=0.1, seed=42)
    
    mu_list = [0.0, 0.001, 0.01, 0.1, 1.0]
    epoch_list = [2, 5]
    
    results = []
    print("\n--- Running FedProx Grid Search on Extreme Silos and Dirichlet 0.1 ---")
    for partition_name, p_data in [("Extreme Silos", silos), ("Dirichlet alpha=0.1", dir_01)]:
        for epochs in epoch_list:
            for mu in mu_list:
                strat_name = "FedAvg" if mu == 0.0 else f"FedProx (mu={mu})"
                m = run_fl_trial(p_data, input_dim, X_test, y_test, num_rounds=15, local_epochs=epochs, proximal_mu=mu, seed=42)
                print(f"[{partition_name}] Epochs={epochs} | {strat_name:20s} -> F1: {m['f1']:.4f}, Recall: {m['recall']:.4f}, Precision: {m['precision']:.4f}, FPR: {m['fpr']*100:.2f}%")
                results.append({
                    "Partition": partition_name,
                    "Local_Epochs": epochs,
                    "Mu": mu,
                    "Strategy": strat_name,
                    "F1": round(m["f1"], 4),
                    "Recall": round(m["recall"], 4),
                    "Precision": round(m["precision"], 4),
                    "FPR": round(m["fpr"], 4)
                })
                
    df_res = pd.DataFrame(results)
    df_res.to_csv(EXPERIMENTS_DIR / "fedprox_sensitivity_grid.csv", index=False)
    print("\nSaved sensitivity results to experiments/fedprox_sensitivity_grid.csv")

if __name__ == "__main__":
    main()
