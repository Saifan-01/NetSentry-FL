"""
NetSentry-FL: Cross-Dataset and Out-of-Distribution (OOD) Domain Generalization Benchmark
Evaluates trained models against distribution shifts representing real-world network variations:
1. Standard In-Distribution Test: Baseline CIC-IDS2017 test set.
2. Covariate WAN Drift: Network congestion, jitter, and packet sizing drift.
3. Adversarial Stealth Mutation: Attack pacing and payload fragmentation mimicry.
"""

import os
import sys
import copy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import Dict, List, Tuple

from src.models.ids_net import NetSentryClassifier
from src.federated.partitioners import create_extreme_attack_silos

SEEDS = [42, 101, 777]
BATCH_SIZE = 128
ROUNDS = 10
LOCAL_EPOCHS = 2
LR = 0.001

def train_centralized(X_tr, y_tr, seed: int) -> NetSentryClassifier:
    input_dim = X_tr.shape[1]
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = NetSentryClassifier(input_dim=input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    dataset = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model.train()
    for _ in range(5):
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
    return model

def simulate_fl(partitions, X_val, y_val, mu: float, seed: int, input_dim: int) -> NetSentryClassifier:
    torch.manual_seed(seed)
    np.random.seed(seed)
    global_model = NetSentryClassifier(input_dim=input_dim)
    criterion = nn.BCEWithLogitsLoss()
    total_samples = sum(len(p["y"]) for p in partitions.values())
    
    for r in range(ROUNDS):
        client_weights = []
        client_sizes = []
        global_params = [p.clone().detach() for p in global_model.parameters()]
        
        for cid, p_data in partitions.items():
            c_model = copy.deepcopy(global_model)
            c_model.train()
            c_opt = torch.optim.Adam(c_model.parameters(), lr=LR)
            c_ds = TensorDataset(torch.tensor(p_data["X"], dtype=torch.float32), torch.tensor(p_data["y"], dtype=torch.float32))
            c_loader = DataLoader(c_ds, batch_size=BATCH_SIZE, shuffle=True)
            
            for _ in range(LOCAL_EPOCHS):
                for bx, by in c_loader:
                    c_opt.zero_grad()
                    out = c_model(bx)
                    loss = criterion(out, by)
                    if mu > 0.0:
                        prox = 0.0
                        for cp, gp in zip(c_model.parameters(), global_params):
                            prox += ((cp - gp) ** 2).sum()
                        loss = loss + (mu / 2.0) * prox
                    loss.backward()
                    c_opt.step()
                    
            client_weights.append([p.detach().clone() for p in c_model.parameters()])
            client_sizes.append(len(p_data["y"]))
            
        new_params = []
        for p_idx in range(len(global_params)):
            avg_p = sum(client_weights[c][p_idx] * (client_sizes[c] / total_samples) for c in range(len(client_sizes)))
            new_params.append(avg_p)
            
        with torch.no_grad():
            for gp, np_t in zip(global_model.parameters(), new_params):
                gp.copy_(np_t)
                
    return global_model

def evaluate_model(model: nn.Module, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        probs = torch.sigmoid(logits).numpy()
        preds = (probs >= 0.5).astype(int)
        
    acc = accuracy_score(y, preds)
    p, r, f1, _ = precision_recall_fscore_support(y, preds, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1, "fpr": fpr}

def run_cross_dataset_benchmark():
    data_dir = "data/processed"
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    train_meta = pd.read_csv(os.path.join(data_dir, "train_metadata.csv"))
    
    # 1. Clean In-Distribution Test
    test_clean = (X_test, y_test)
    
    # 2. Shift A: Covariate WAN Drift (jitter & packet sizing perturbation)
    np.random.seed(42)
    X_shift_a = X_test.copy()
    noise = np.random.normal(loc=0.0, scale=0.25, size=X_test.shape)
    # Apply perturbation to flow rates and IAT features (features 10 to 45)
    X_shift_a[:, 10:45] += noise[:, 10:45]
    test_shift_a = (X_shift_a, y_test)
    
    # 3. Shift B: Adversarial Stealth Mutation (attack mimicry)
    X_shift_b = X_test.copy()
    attack_mask = (y_test == 1)
    # Attacker stealth: dampening burst rates towards normal distribution
    X_shift_b[attack_mask, :20] *= 0.65
    test_shift_b = (X_shift_b, y_test)
    
    regimes = {
        "In-Distribution": test_clean,
        "Covariate WAN Drift": test_shift_a,
        "Stealth Attack Mutation": test_shift_b
    }
    
    records = []
    
    input_dim = X_train.shape[1]
    for seed in SEEDS:
        print(f"\nEvaluating Seed {seed}...")
        # Train Centralized
        cent_model = train_centralized(X_train, y_train, seed=seed)
        
        # Partition Silos
        silo_partitions = create_extreme_attack_silos(X_train, train_meta, seed=seed)
        
        # Train FedAvg (Silos)
        fedavg_model = simulate_fl(silo_partitions, X_val, y_val, mu=0.0, seed=seed, input_dim=input_dim)
        
        # Train FedProx (Silos, mu=1.0)
        fedprox_model = simulate_fl(silo_partitions, X_val, y_val, mu=1.0, seed=seed, input_dim=input_dim)
        
        # Dynamic INT8 Quantized FedProx Model
        q_model = torch.ao.quantization.quantize_dynamic(
            copy.deepcopy(fedprox_model),
            {nn.Linear},
            dtype=torch.qint8
        )
        
        models = {
            "Centralized MLP": cent_model,
            "FedAvg (Silos)": fedavg_model,
            "FedProx (mu=1.0, Silos)": fedprox_model,
            "INT8 Quantized FedProx": q_model
        }
        
        for m_name, mdl in models.items():
            for r_name, (X_eval, y_eval) in regimes.items():
                m = evaluate_model(mdl, X_eval, y_eval)
                records.append({
                    "seed": seed,
                    "model": m_name,
                    "regime": r_name,
                    "accuracy": m["accuracy"],
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "f1": m["f1"],
                    "fpr": m["fpr"]
                })
                
    df = pd.DataFrame(records)
    os.makedirs("experiments", exist_ok=True)
    raw_path = "experiments/cross_dataset_results_raw.csv"
    df.to_csv(raw_path, index=False)
    
    # Summary
    summary = df.groupby(["model", "regime"]).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        fpr_mean=("fpr", "mean"),
        fpr_std=("fpr", "std")
    ).reset_index()
    
    summary_path = "experiments/cross_dataset_results.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved cross-dataset summary to {summary_path}")
    print(summary.to_string())

if __name__ == "__main__":
    run_cross_dataset_benchmark()
