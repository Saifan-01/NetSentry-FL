"""
NetSentry-FL: Client Data Partitioner
Simulates multi-organization data silos across 3 clients:
- Client 0 (Organization A - Financial Institution)
- Client 1 (Organization B - Healthcare Provider)
- Client 2 (Organization C - Cloud Infrastructure)

Supports both:
1. IID (Independent and Identically Distributed)
2. Non-IID (Realistic cross-organizational data heterogeneity)
"""

import numpy as np
from typing import Dict, Tuple, List
from pathlib import Path

PROCESSED_DIR = Path("data/processed")

def create_client_partitions(
    num_clients: int = 3,
    non_iid: bool = False,
    random_state: int = 42
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Splits the training data across num_clients simulated organizations.
    Returns:
        dict: {client_id: {"X": np.ndarray, "y": np.ndarray}}
    """
    np.random.seed(random_state)
    X_train = np.load(PROCESSED_DIR / "X_train.npy").astype(np.float32)
    y_train = np.load(PROCESSED_DIR / "y_train.npy").astype(np.float32)
    
    n_samples = len(X_train)
    client_data = {}
    
    if not non_iid:
        # IID: Uniform random permutation across clients
        indices = np.random.permutation(n_samples)
        splits = np.array_split(indices, num_clients)
        
        for cid in range(num_clients):
            idx = splits[cid]
            client_data[cid] = {
                "X": X_train[idx],
                "y": y_train[idx]
            }
            benign_cnt = (y_train[idx] == 0).sum()
            attack_cnt = (y_train[idx] == 1).sum()
            print(f"[*] Client {cid} (IID) Assigned {len(idx)} flows: Benign={benign_cnt}, Attack={attack_cnt}")
    else:
        # Non-IID: Skewed attack-to-benign ratios across clients
        benign_idx = np.where(y_train == 0)[0]
        attack_idx = np.where(y_train == 1)[0]
        
        np.random.shuffle(benign_idx)
        np.random.shuffle(attack_idx)
        
        # Split benign evenly
        benign_splits = np.array_split(benign_idx, num_clients)
        
        # Split attacks non-identically (e.g. 60% client 0, 30% client 1, 10% client 2)
        n_att = len(attack_idx)
        cut1 = int(n_att * 0.60)
        cut2 = int(n_att * 0.90)
        attack_splits = [
            attack_idx[:cut1],
            attack_idx[cut1:cut2],
            attack_idx[cut2:]
        ]
        
        for cid in range(num_clients):
            idx = np.concatenate([benign_splits[cid], attack_splits[cid]])
            np.random.shuffle(idx)
            client_data[cid] = {
                "X": X_train[idx],
                "y": y_train[idx]
            }
            benign_cnt = (y_train[idx] == 0).sum()
            attack_cnt = (y_train[idx] == 1).sum()
            print(f"[*] Client {cid} (Non-IID) Assigned {len(idx)} flows: Benign={benign_cnt}, Attack={attack_cnt} ({attack_cnt/len(idx)*100:.1f}% attack rate)")

    return client_data

if __name__ == "__main__":
    print("Testing IID Partitions:")
    create_client_partitions(num_clients=3, non_iid=False)
    print("\nTesting Non-IID Partitions:")
    create_client_partitions(num_clients=3, non_iid=True)
