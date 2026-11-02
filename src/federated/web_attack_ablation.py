"""
NetSentry-FL: Web Attack Minority Class Ablation Study
Investigates the cause of Web Attack recall collapse under Extreme Silos in FedAvg:
1. Standard FedAvg (Sample-Weighted: w_k = n_k / N)
2. Equal Client-Weighted FedAvg (Uniform: w_k = 1 / K)
3. FedProx (mu = 0.01, Sample-Weighted)

Averaged across 5 seeds: [42, 101, 777, 1234, 5678].
Logs round-by-round recall on Web Attacks (429 test flows) and validation set metrics.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from src.models.ids_net import NetSentryClassifier, get_model_parameters, set_model_parameters
from src.federated.client import NetSentryClient
from src.federated.partitioners import create_extreme_attack_silos

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

def aggregate_weights(updates, weighting_type="sample_weighted"):
    """
    weighting_type: 'sample_weighted' (w_k = n_k / N) or 'equal_weighted' (w_k = 1 / K)
    """
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

def run_ablation_simulation(
    extreme_silos, input_dim, X_val, y_val, val_meta, X_test, y_test, test_meta,
    weighting_type="sample_weighted", proximal_mu=0.0, num_rounds=15, seed=42
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    clients = [
        NetSentryClient(cid, extreme_silos[cid]["X"], extreme_silos[cid]["y"], input_dim)
        for cid in range(3)
    ]
    global_model = NetSentryClassifier(input_dim=input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.1)
    global_params = get_model_parameters(global_model)
    
    config = {
        "local_epochs": 2,
        "batch_size": 64,
        "lr": 0.001,
        "proximal_mu": proximal_mu
    }
    
    history = []
    web_idx_test = test_meta[test_meta["Category"] == "Web Attack"].index
    
    for rnd in range(1, num_rounds + 1):
        updates = []
        for client in clients:
            up, num, _ = client.fit(global_params, config)
            updates.append((up, num, {}))
            
        global_params = aggregate_weights(updates, weighting_type=weighting_type)
        set_model_parameters(global_model, global_params)
        
        global_model.eval()
        with torch.no_grad():
            test_probs = torch.sigmoid(global_model(torch.from_numpy(X_test))).numpy()
            val_probs = torch.sigmoid(global_model(torch.from_numpy(X_val))).numpy()
            
        test_preds = (test_probs >= 0.5).astype(int)
        val_preds = (val_probs >= 0.5).astype(int)
        
        web_rec = float(test_preds[web_idx_test].mean())
        test_f1 = f1_score(y_test, test_preds, zero_division=0)
        val_f1 = f1_score(y_val, val_preds, zero_division=0)
        
        history.append({
            "seed": seed,
            "round": rnd,
            "val_f1": val_f1,
            "test_f1": test_f1,
            "web_attack_recall": web_rec
        })
        
    return global_model, test_probs, val_probs, history

def main():
    seeds = [42, 101, 777, 1234, 5678]
    print(f"[*] Running Web Attack Collapse Ablation across {len(seeds)} seeds...", flush=True)
    
    X_train = np.load(PROCESSED_DIR / "X_train.npy").astype(np.float32)
    X_val = np.load(PROCESSED_DIR / "X_val.npy").astype(np.float32)
    y_val = np.load(PROCESSED_DIR / "y_val.npy").astype(np.float32)
    X_test = np.load(PROCESSED_DIR / "X_test.npy").astype(np.float32)
    y_test = np.load(PROCESSED_DIR / "y_test.npy").astype(np.float32)
    
    train_meta = pd.read_csv(PROCESSED_DIR / "train_metadata.csv")
    val_meta = pd.read_csv(PROCESSED_DIR / "val_metadata.csv")
    test_meta = pd.read_csv(PROCESSED_DIR / "test_metadata.csv")
    input_dim = X_train.shape[1]
    
    ablation_setups = [
        {"name": "FedAvg (Sample-Weighted)", "weighting": "sample_weighted", "mu": 0.0},
        {"name": "FedAvg (Equal-Weighted 1/K)", "weighting": "equal_weighted", "mu": 0.0},
        {"name": "FedProx (mu=0.01, Sample-Weighted)", "weighting": "sample_weighted", "mu": 0.01}
    ]
    
    results_summary = []
    round_records = []
    
    for setup in ablation_setups:
        sname = setup["name"]
        print(f"\n--- Testing Strategy: {sname} ---", flush=True)
        final_test_f1s = []
        final_web_recalls = []
        final_val_f1s = []
        
        for seed in seeds:
            silos = create_extreme_attack_silos(X_train, train_meta, seed=seed)
            _, t_probs, v_probs, hist = run_ablation_simulation(
                silos, input_dim, X_val, y_val, val_meta, X_test, y_test, test_meta,
                weighting_type=setup["weighting"], proximal_mu=setup["mu"], num_rounds=15, seed=seed
            )
            for h in hist:
                h["strategy"] = sname
                round_records.append(h)
                
            last_record = hist[-1]
            final_test_f1s.append(last_record["test_f1"])
            final_web_recalls.append(last_record["web_attack_recall"])
            final_val_f1s.append(last_record["val_f1"])
            print(f"  [Seed {seed}] Val F1: {last_record['val_f1']:.4f} | Test F1: {last_record['test_f1']:.4f} | Web Attack Recall: {last_record['web_attack_recall']*100:.2f}%", flush=True)
            
        results_summary.append({
            "Strategy": sname,
            "Val_F1_Mean": np.mean(final_val_f1s),
            "Val_F1_Std": np.std(final_val_f1s),
            "Test_F1_Mean": np.mean(final_test_f1s),
            "Test_F1_Std": np.std(final_test_f1s),
            "Web_Recall_Mean": np.mean(final_web_recalls),
            "Web_Recall_Std": np.std(final_web_recalls),
            "Test_F1_Display": f"{np.mean(final_test_f1s):.4f} +/- {np.std(final_test_f1s):.4f}",
            "Web_Recall_Display": f"{np.mean(final_web_recalls)*100:.2f}% +/- {np.std(final_web_recalls)*100:.2f}%"
        })
        
    df_sum = pd.DataFrame(results_summary)
    df_sum.to_csv(EXPERIMENTS_DIR / "web_attack_ablation_summary.csv", index=False)
    pd.DataFrame(round_records).to_csv(EXPERIMENTS_DIR / "web_attack_ablation_history.csv", index=False)
    
    print("\n" + "="*85, flush=True)
    print("             WEB ATTACK MINORITY CLASS ABLATION (5 SEEDS)             ", flush=True)
    print("="*85, flush=True)
    print(df_sum[["Strategy", "Test_F1_Display", "Web_Recall_Display"]].to_string(index=False), flush=True)
    print("="*85 + "\n", flush=True)

if __name__ == "__main__":
    main()
