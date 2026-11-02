import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from src.federated.partitioners import create_extreme_attack_silos, create_dirichlet_partitions
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

X_train = np.load(PROCESSED_DIR / "X_train.npy")
train_meta = pd.read_csv(PROCESSED_DIR / "train_metadata.csv")

print("=== 1. EXTREME ATTACK SILOS (Seed 42) ===")
silos = create_extreme_attack_silos(X_train, train_meta, seed=42)
silo_names = ["Client A (DoS Silo)", "Client B (PortScan Silo)", "Client C (Web Attack Silo)"]
for cid, data in silos.items():
    cats = pd.Series(data["categories"]).value_counts().to_dict()
    b_count = int((data["y"] == 0).sum())
    a_count = int((data["y"] == 1).sum())
    print(f"Client {cid} ({silo_names[cid]}): Total={len(data['y'])}, Benign={b_count}, Attack={a_count}, Attack%={a_count/len(data['y'])*100:.2f}%, Breakdown={cats}")

print("\n=== 2. DIRICHLET (alpha = 0.5, Seed 42) ===")
dir_05 = create_dirichlet_partitions(X_train, train_meta, alpha=0.5, seed=42)
for cid, data in dir_05.items():
    cats = pd.Series(data["categories"]).value_counts().to_dict()
    b_count = int((data["y"] == 0).sum())
    a_count = int((data["y"] == 1).sum())
    print(f"Client {cid}: Total={len(data['y'])}, Benign={b_count}, Attack={a_count}, Attack%={a_count/len(data['y'])*100:.2f}%, Breakdown={cats}")

print("\n=== 3. DIRICHLET (alpha = 0.1, Seed 42) ===")
dir_01 = create_dirichlet_partitions(X_train, train_meta, alpha=0.1, seed=42)
for cid, data in dir_01.items():
    cats = pd.Series(data["categories"]).value_counts().to_dict()
    b_count = int((data["y"] == 0).sum())
    a_count = int((data["y"] == 1).sum())
    print(f"Client {cid}: Total={len(data['y'])}, Benign={b_count}, Attack={a_count}, Attack%={a_count/len(data['y'])*100:.2f}%, Breakdown={cats}")
