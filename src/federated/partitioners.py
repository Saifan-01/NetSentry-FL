"""
NetSentry-FL: Advanced Multi-Organization Partitioner (Month 5)
Provides:
1. Extreme Attack Silo Partitions (Client A = DoS, Client B = PortScan, Client C = Web Attack + Benign)
2. Continuous Dirichlet Label-Skew Partitions: Dir(alpha) for alpha = 0.5 and alpha = 0.1
3. Distribution Visualization Generator: Produces publication plots of client class proportions.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHARTS_DIR = PROJECT_ROOT / "reports" / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

def create_extreme_attack_silos(X_train: np.ndarray, train_meta: pd.DataFrame, seed: int = 42) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Simulates extreme cross-organizational data silos:
    - Client 0 (FinTech / Bank): Encounters exclusively DoS attacks + Benign
    - Client 1 (Healthcare System): Encounters exclusively PortScan attacks + Benign
    - Client 2 (Cloud Infrastructure): Encounters exclusively Web Attacks + Benign
    Benign flows are split equally across clients to simulate normal background operations.
    """
    np.random.seed(seed)
    categories = train_meta["Category"].values
    binary_labels = train_meta["Binary_Label"].values
    
    benign_idx = np.where(categories == "BENIGN")[0]
    dos_idx = np.where(categories == "DoS")[0]
    ps_idx = np.where(categories == "PortScan")[0]
    web_idx = np.where(categories == "Web Attack")[0]
    
    np.random.shuffle(benign_idx)
    b_splits = np.array_split(benign_idx, 3)
    
    client_indices = {
        0: np.concatenate([b_splits[0], dos_idx]),
        1: np.concatenate([b_splits[1], ps_idx]),
        2: np.concatenate([b_splits[2], web_idx])
    }
    
    silos = {}
    for cid in range(3):
        idx = client_indices[cid]
        np.random.shuffle(idx)
        silos[cid] = {
            "X": X_train[idx],
            "y": binary_labels[idx].astype(np.float32),
            "categories": categories[idx]
        }
    return silos

def create_dirichlet_partitions(
    X_train: np.ndarray,
    train_meta: pd.DataFrame,
    alpha: float = 0.5,
    num_clients: int = 3,
    seed: int = 42
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Partitions training data among num_clients using a Dirichlet distribution Dir(alpha).
    - Lower alpha (e.g. 0.1) -> Severe label skew (heterogeneous).
    - Higher alpha (e.g. 0.5) -> Moderate label skew.
    - alpha -> inf -> Uniform IID.
    Guarantees no client receives an empty partition.
    """
    np.random.seed(seed)
    categories = train_meta["Category"].values
    binary_labels = train_meta["Binary_Label"].values
    unique_classes = sorted(train_meta["Category"].unique())
    
    client_idx_map = {cid: [] for cid in range(num_clients)}
    min_per_client = 15
    
    for c in unique_classes:
        c_idx = np.where(categories == c)[0]
        np.random.shuffle(c_idx)
        
        # Guarantee non-zero class representation per client to avoid degenerate single-class batches
        floor_count = min(min_per_client, len(c_idx) // (num_clients * 2))
        reserved_indices = []
        for cid in range(num_clients):
            start = cid * floor_count
            end = (cid + 1) * floor_count
            client_idx_map[cid].append(c_idx[start:end])
            
        remaining_idx = c_idx[num_clients * floor_count:]
        
        # Sample Dirichlet distribution vector for the remaining majority of samples
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        cutoffs = (np.cumsum(proportions)[:-1] * len(remaining_idx)).astype(int)
        splits = np.split(remaining_idx, cutoffs)
        
        for cid in range(num_clients):
            client_idx_map[cid].append(splits[cid])
            
    partitions = {}
    for cid in range(num_clients):
        all_idx = np.concatenate(client_idx_map[cid])
        np.random.shuffle(all_idx)
        partitions[cid] = {
            "X": X_train[all_idx],
            "y": binary_labels[all_idx].astype(np.float32),
            "categories": categories[all_idx]
        }
    return partitions

def plot_client_class_distributions(silos_dict: Dict[str, Dict[int, Dict[str, Any]]], save_path: Path):
    """
    Generates a clear publication visualization comparing class distributions across:
    1. Extreme Silos
    2. Dirichlet alpha=0.5
    3. Dirichlet alpha=0.1
    """
    plt.style.use("tableau-colorblind10")
    fig, axes = plt.subplots(1, len(silos_dict), figsize=(15, 4.5), sharey=True)
    
    class_order = ["BENIGN", "DoS", "PortScan", "Web Attack"]
    colors = ["#2ca02c", "#d62728", "#1f77b4", "#ff7f0e"]
    
    for ax_idx, (split_name, partitions) in enumerate(silos_dict.items()):
        ax = axes[ax_idx] if len(silos_dict) > 1 else axes
        data_to_plot = []
        for cid in range(3):
            cats = partitions[cid]["categories"]
            counts = {c: (cats == c).sum() for c in class_order}
            data_to_plot.append(counts)
            
        df_plot = pd.DataFrame(data_to_plot, index=["Client A", "Client B", "Client C"])
        df_plot.plot(kind="bar", stacked=True, ax=ax, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(split_name, fontsize=11, weight="bold", pad=8)
        ax.set_xlabel("Simulated Organization", fontsize=10)
        if ax_idx == 0:
            ax.set_ylabel("Number of Flow Records", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.tick_params(axis="x", rotation=0)
        if ax_idx == len(silos_dict) - 1:
            ax.legend(title="Traffic Category", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
        else:
            if ax.get_legend() is not None:
                ax.get_legend().remove()
                
    plt.suptitle("Client Data Heterogeneity: Extreme Silos vs. Dirichlet Label Skew", fontsize=13, weight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved client distribution plot to {save_path}")

if __name__ == "__main__":
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    X_train = np.load(PROCESSED_DIR / "X_train.npy").astype(np.float32)
    train_meta = pd.read_csv(PROCESSED_DIR / "train_metadata.csv")
    
    silo_extreme = create_extreme_attack_silos(X_train, train_meta, seed=42)
    silo_dir_05 = create_dirichlet_partitions(X_train, train_meta, alpha=0.5, seed=42)
    silo_dir_01 = create_dirichlet_partitions(X_train, train_meta, alpha=0.1, seed=42)
    
    plot_dict = {
        "Extreme Attack Silos": silo_extreme,
        "Dirichlet (alpha = 0.5)": silo_dir_05,
        "Dirichlet (alpha = 0.1)": silo_dir_01
    }
    plot_client_class_distributions(plot_dict, CHARTS_DIR / "dirichlet_and_silo_distributions.png")
