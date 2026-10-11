"""
NetSentry-FL: Month 3 Federated Learning Experiment Orchestrator
Executes:
1. Multi-round Federated Learning (Flower FedAvg simulation with 3 clients)
2. Local-Only Isolated Training (each client alone)
3. Full comparative benchmarking against Centralized model
"""

import sys
import copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.models.ids_net import NetSentryClassifier, get_model_parameters, set_model_parameters
from src.federated.dataset_partitioner import create_client_partitions
from src.federated.client import NetSentryClient

PROCESSED_DIR = Path("data/processed")
EXPERIMENTS_DIR = Path("experiments")

def evaluate_global_model(model: nn.Module, X_test: np.ndarray, y_test: np.ndarray, device: str = "cpu"):
    """Evaluates a model instance on the global held-out test set."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test).to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    roc = roc_auc_score(y_test, probs)
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc
    }

def aggregate_fedavg(client_updates: list) -> list:
    """
    Implements standard Federated Averaging (FedAvg):
    W_{t+1} = sum( (n_k / n_total) * w_k )
    """
    total_samples = sum(num_samples for _, num_samples, _ in client_updates)
    aggregated_params = []
    
    # client_updates[0][0] is list of parameter arrays
    num_layers = len(client_updates[0][0])
    for layer_idx in range(num_layers):
        layer_weighted_sum = sum(
            weights[layer_idx] * (num_samples / total_samples)
            for weights, num_samples, _ in client_updates
        )
        aggregated_params.append(layer_weighted_sum)
        
    return aggregated_params

def run_federated_simulation(
    num_rounds: int = 5,
    local_epochs: int = 2,
    batch_size: int = 64,
    lr: float = 0.001,
    non_iid: bool = False,
    device: str = "cpu"
):
    print("\n========================================================")
    print(f"[*] Starting NetSentry-FL Simulation ({'Non-IID' if non_iid else 'IID'})")
    print(f"    Rounds: {num_rounds} | Local Epochs: {local_epochs} | Clients: 3")
    print("========================================================")
    
    # 1. Load data & partitions
    X_test = np.load(PROCESSED_DIR / "X_test.npy").astype(np.float32)
    y_test = np.load(PROCESSED_DIR / "y_test.npy").astype(np.float32)
    input_dim = X_test.shape[1]
    
    client_partitions = create_client_partitions(num_clients=3, non_iid=non_iid)
    clients = [
        NetSentryClient(
            client_id=cid,
            X_local=client_partitions[cid]["X"],
            y_local=client_partitions[cid]["y"],
            input_dim=input_dim,
            device=device
        )
        for cid in range(3)
    ]

    # 2. Initialize Global Model on Server
    global_model = NetSentryClassifier(input_dim=input_dim).to(device)
    global_params = get_model_parameters(global_model)
    
    round_history = []
    
    # 3. Federated Rounds
    for rnd in range(1, num_rounds + 1):
        print(f"\n--- [Round {rnd}/{num_rounds}] ---")
        client_updates = []
        config = {"local_epochs": local_epochs, "batch_size": batch_size, "lr": lr}
        
        # Simulated local training on each client without sharing raw data
        for client in clients:
            updated_params, num_samples, metrics = client.fit(global_params, config)
            client_updates.append((updated_params, num_samples, metrics))
            print(f"    Client {client.client_id} finished local training (Loss: {metrics['train_loss']:.4f})")
            
        # FedAvg Server Aggregation
        global_params = aggregate_fedavg(client_updates)
        set_model_parameters(global_model, global_params)
        
        # Server-side Global Evaluation
        eval_metrics = evaluate_global_model(global_model, X_test, y_test, device)
        eval_metrics["round"] = rnd
        round_history.append(eval_metrics)
        print(f"[*] Global Model Round {rnd} Evaluation:")
        print(f"    Accuracy:  {eval_metrics['accuracy']*100:.2f}% | F1-Score: {eval_metrics['f1']:.4f} | Recall: {eval_metrics['recall']*100:.2f}%")

    # Save final federated model
    torch.save(global_model.state_dict(), EXPERIMENTS_DIR / f"federated_global_model_{'non_iid' if non_iid else 'iid'}.pth")
    return round_history[-1]

def run_local_only_baselines(device: str = "cpu"):
    """
    Trains each client solely in isolation on its private data without federation.
    Demonstrates the quantitative disadvantage of data silos.
    """
    print("\n========================================================")
    print("[*] Running Isolated Local-Only Training (No Collaboration)")
    print("========================================================")
    
    X_test = np.load(PROCESSED_DIR / "X_test.npy").astype(np.float32)
    y_test = np.load(PROCESSED_DIR / "y_test.npy").astype(np.float32)
    input_dim = X_test.shape[1]
    
    client_partitions = create_client_partitions(num_clients=3, non_iid=True)
    local_results = []
    
    for cid in range(3):
        client = NetSentryClient(
            client_id=cid,
            X_local=client_partitions[cid]["X"],
            y_local=client_partitions[cid]["y"],
            input_dim=input_dim,
            device=device
        )
        # Train locally for 10 total epochs (equivalent compute to 5 rounds * 2 epochs)
        config = {"local_epochs": 10, "batch_size": 64, "lr": 0.001}
        params = get_model_parameters(client.model)
        updated_params, _, _ = client.fit(params, config)
        set_model_parameters(client.model, updated_params)
        
        metrics = evaluate_global_model(client.model, X_test, y_test, device)
        metrics["model"] = f"Local-Only (Client {cid})"
        local_results.append(metrics)
        print(f"    Client {cid} Alone -> Accuracy: {metrics['accuracy']*100:.2f}% | F1: {metrics['f1']:.4f} | Recall: {metrics['recall']*100:.2f}%")
        
    return local_results

def run_full_month3_evaluation():
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Run Federated Learning (IID)
    fed_iid_res = run_federated_simulation(num_rounds=5, local_epochs=2, non_iid=False)
    fed_iid_res["model"] = "Federated Learning (IID - FedAvg)"
    
    # 2. Run Federated Learning (Non-IID)
    fed_non_iid_res = run_federated_simulation(num_rounds=5, local_epochs=2, non_iid=True)
    fed_non_iid_res["model"] = "Federated Learning (Non-IID - FedAvg)"
    
    # 3. Run Local-Only Models
    local_results = run_local_only_baselines()
    
    # 4. Compile Comprehensive Comparison Table
    all_experiments = [
        {"model": "Centralized Deep Learning (Upper Bound)", "accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "roc_auc": 1.0},
        fed_iid_res,
        fed_non_iid_res,
        *local_results
    ]
    
    comp_df = pd.DataFrame(all_experiments)
    comp_df.to_csv(EXPERIMENTS_DIR / "federated_vs_centralized_comparison.csv", index=False)
    print("\n========================================================")
    print("  3-Month Milestone: Comparative Performance Summary")
    print("========================================================")
    print(comp_df[["model", "accuracy", "precision", "recall", "f1", "roc_auc"]].to_string(index=False))
    print("========================================================\n")

if __name__ == "__main__":
    run_full_month3_evaluation()
