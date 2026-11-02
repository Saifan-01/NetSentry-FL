"""
NetSentry-FL: Differential Privacy Evaluation via Opacus (DP-SGD)
Evaluates the privacy-utility tradeoff for Federated Learning on CIC-IDS2017:
- Per-client DP-SGD using Opacus PrivacyEngine
- Noise multiplier sweep: sigma in {0.5, 1.0, 2.0} with fixed clipping norm C = 1.0
- Target delta = 1e-5 (strict delta < 1/N for N=25,522)
- Architecture: GroupNorm MLP (opacus-compliant, batch-size independent)
- Primary benchmark: Federated IID (clean epsilon-vs-F1 trade-off curve)
- Heterogeneous benchmark: Extreme Silos with FedProx (mu*=1.0)
- Fair comparison: Non-DP baseline executed with the EXACT same GroupNorm architecture and scaling!

Evaluated across all 5 seeds: [42, 101, 777, 1234, 5678].
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os
import io
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, precision_recall_curve, auc
from opacus import PrivacyEngine

from src.models.ids_net import NetSentryClassifier, get_model_parameters, set_model_parameters
from src.federated.partitioners import create_extreme_attack_silos

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHARTS_DIR = PROJECT_ROOT / "reports" / "charts"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

def evaluate_metrics(y_true: np.ndarray, y_prob: np.ndarray):
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    p_curve, r_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(r_curve, p_curve)
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(pr_auc),
        "fpr": float(fpr)
    }

def aggregate_weights(updates):
    total_samples = sum(num for _, num, _ in updates)
    weights = [num / total_samples for _, num, _ in updates]
    num_layers = len(updates[0][0])
    avg_params = []
    for layer_idx in range(num_layers):
        layer_avg = sum(w * up[layer_idx] for (up, _, _), w in zip(updates, weights))
        avg_params.append(layer_avg)
    return avg_params

class DPFederatedClient:
    def __init__(self, cid: int, X: np.ndarray, y: np.ndarray, input_dim: int,
                 enable_dp: bool = False, noise_multiplier: float = 1.0, max_grad_norm: float = 1.0):
        self.cid = cid
        self.enable_dp = enable_dp
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        
        self.model = NetSentryClassifier(input_dim=input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.1, norm_type="group_norm")
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        
        pos_samples = float(y.sum())
        neg_samples = float(len(y) - pos_samples)
        pos_weight = torch.tensor([min(neg_samples / max(pos_samples, 1.0), 10.0)])
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        self.num_samples = len(X)
        self.dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
        # In Opacus, DPDataLoader doesn't support drop_last=True
        self.loader = DataLoader(self.dataset, batch_size=128, shuffle=True)
        
        self.privacy_engine = None
        if self.enable_dp:
            self.privacy_engine = PrivacyEngine(secure_mode=False)
            self.model, self.optimizer, self.loader = self.privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.loader,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm
            )

    def fit(self, global_params, local_epochs=2, proximal_mu=0.0):
        # Set parameters on the underlying module
        target_mod = self.model._module if self.enable_dp else self.model
        set_model_parameters(target_mod, global_params)
        
        global_tensors = []
        if proximal_mu > 0.0:
            for p in target_mod.parameters():
                global_tensors.append(p.detach().clone())
                
        self.model.train()
        for _ in range(local_epochs):
            for bx, by in self.loader:
                self.optimizer.zero_grad()
                logits = self.model(bx)
                loss = self.criterion(logits, by)
                
                if proximal_mu > 0.0:
                    p_term = sum(torch.sum((p - gp) ** 2) for p, gp in zip(target_mod.parameters(), global_tensors))
                    loss += (proximal_mu / 2.0) * p_term
                    
                loss.backward()
                self.optimizer.step()
                
        # Return updated parameters as list of numpy arrays
        updated_params = [p.detach().cpu().numpy() for p in target_mod.parameters()]
        return updated_params, self.num_samples

    def get_epsilon(self, delta=1e-5):
        if self.enable_dp and self.privacy_engine is not None:
            return float(self.privacy_engine.get_epsilon(delta=delta))
        return float("inf")

def run_dp_simulation(clients_dict, input_dim, X_test, y_test, test_meta,
                       num_rounds=15, local_epochs=2, proximal_mu=0.0,
                       enable_dp=False, noise_multiplier=1.0, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    clients = [
        DPFederatedClient(cid, clients_dict[cid]["X"], clients_dict[cid]["y"], input_dim,
                           enable_dp=enable_dp, noise_multiplier=noise_multiplier, max_grad_norm=1.0)
        for cid in range(len(clients_dict))
    ]
    global_model = NetSentryClassifier(input_dim=input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.1, norm_type="group_norm")
    global_params = get_model_parameters(global_model)
    
    for rnd in range(1, num_rounds + 1):
        updates = []
        for client in clients:
            up, num = client.fit(global_params, local_epochs=local_epochs, proximal_mu=proximal_mu)
            updates.append((up, num, {}))
        global_params = aggregate_weights(updates)
        set_model_parameters(global_model, global_params)
        
    # Evaluate global model
    global_model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(global_model(torch.from_numpy(X_test))).numpy()
    m = evaluate_metrics(y_test, probs)
    
    # Per-client epsilon (average across clients)
    eps_vals = [c.get_epsilon(delta=1e-5) for c in clients]
    mean_eps = float(np.mean(eps_vals))
    
    # Per-attack recall
    preds = (probs >= 0.5).astype(int)
    att_recalls = {}
    for att in ["DoS", "PortScan", "Web Attack"]:
        idx = test_meta[test_meta["Category"] == att].index
        att_recalls[f"{att}_Recall"] = float(preds[idx].mean())
        
    return m, mean_eps, att_recalls

def main():
    import os
    torch.set_num_threads(min(12, os.cpu_count() or 4))
    print("="*95, flush=True)
    print("      NETSENTRY-FL: MONTH 6 DIFFERENTIAL PRIVACY BENCHMARK (OPACUS DP-SGD)      ", flush=True)
    print("="*95, flush=True)
    
    X_train = np.load(PROCESSED_DIR / "X_train.npy").astype(np.float32)
    y_train = np.load(PROCESSED_DIR / "y_train.npy").astype(np.float32)
    X_test = np.load(PROCESSED_DIR / "X_test.npy").astype(np.float32)
    y_test = np.load(PROCESSED_DIR / "y_test.npy").astype(np.float32)
    train_meta = pd.read_csv(PROCESSED_DIR / "train_metadata.csv")
    test_meta = pd.read_csv(PROCESSED_DIR / "test_metadata.csv")
    input_dim = X_train.shape[1]
    
    seeds = [42, 101, 777, 1234, 5678]
    
    # Configs to evaluate:
    configs = [
        {"name": "Non-DP Baseline (GroupNorm, IID)", "partition": "iid", "enable_dp": False, "sigma": 0.0, "mu": 0.0},
        {"name": "DP-SGD (sigma=0.5, IID)", "partition": "iid", "enable_dp": True, "sigma": 0.5, "mu": 0.0},
        {"name": "DP-SGD (sigma=1.0, IID)", "partition": "iid", "enable_dp": True, "sigma": 1.0, "mu": 0.0},
        {"name": "DP-SGD (sigma=2.0, IID)", "partition": "iid", "enable_dp": True, "sigma": 2.0, "mu": 0.0},
        {"name": "Non-DP Baseline (GroupNorm, Silos FedProx mu=1.0)", "partition": "silos", "enable_dp": False, "sigma": 0.0, "mu": 1.0},
        {"name": "DP-SGD (sigma=1.0, Silos FedProx mu=1.0)", "partition": "silos", "enable_dp": True, "sigma": 1.0, "mu": 1.0}
    ]
    
    summary_records = []
    
    for cfg in configs:
        cname = cfg["name"]
        print(f"\n>>> Running Evaluation: {cname} <<<", flush=True)
        run_metrics = []
        eps_list = []
        dos_recalls = []
        ps_recalls = []
        web_recalls = []
        
        for s in seeds:
            if cfg["partition"] == "iid":
                idx_perm = np.random.RandomState(s).permutation(len(X_train))
                splits = np.array_split(idx_perm, 3)
                cdict = {cid: {"X": X_train[splits[cid]], "y": y_train[splits[cid]]} for cid in range(3)}
            else:
                cdict = create_extreme_attack_silos(X_train, train_meta, seed=s)
                
            m, eps, att_r = run_dp_simulation(
                cdict, input_dim, X_test, y_test, test_meta,
                num_rounds=15, local_epochs=2, proximal_mu=cfg["mu"],
                enable_dp=cfg["enable_dp"], noise_multiplier=cfg["sigma"], seed=s
            )
            run_metrics.append(m)
            eps_list.append(eps)
            dos_recalls.append(att_r["DoS_Recall"])
            ps_recalls.append(att_r["PortScan_Recall"])
            web_recalls.append(att_r["Web Attack_Recall"])
            print(f"  [Seed {s}] F1: {m['f1']:.4f} | Recall: {m['recall']:.4f} | Precision: {m['precision']:.4f} | Eps: {eps:.2f}", flush=True)
            
        f1_vals = [r["f1"] for r in run_metrics]
        rec_vals = [r["recall"] for r in run_metrics]
        prec_vals = [r["precision"] for r in run_metrics]
        fpr_vals = [r["fpr"] for r in run_metrics]
        
        summary_records.append({
            "Configuration": cname,
            "Partition": cfg["partition"],
            "DP_Enabled": cfg["enable_dp"],
            "Noise_Multiplier_sigma": cfg["sigma"],
            "Clipping_Norm_C": 1.0 if cfg["enable_dp"] else None,
            "Epsilon_Mean": round(float(np.mean(eps_list)), 2) if cfg["enable_dp"] else None,
            "Delta": 1e-5 if cfg["enable_dp"] else None,
            "F1_Mean": round(float(np.mean(f1_vals)), 4),
            "F1_Std": round(float(np.std(f1_vals)), 4),
            "Recall_Mean": round(float(np.mean(rec_vals)), 4),
            "Recall_Std": round(float(np.std(rec_vals)), 4),
            "Precision_Mean": round(float(np.mean(prec_vals)), 4),
            "Precision_Std": round(float(np.std(prec_vals)), 4),
            "FPR_Mean": round(float(np.mean(fpr_vals)), 4),
            "FPR_Std": round(float(np.std(fpr_vals)), 4),
            "DoS_Recall_Mean": round(float(np.mean(dos_recalls)), 4),
            "PortScan_Recall_Mean": round(float(np.mean(ps_recalls)), 4),
            "Web_Attack_Recall_Mean": round(float(np.mean(web_recalls)), 4),
            "F1": f"{np.mean(f1_vals):.4f} +/- {np.std(f1_vals):.4f}",
            "Recall": f"{np.mean(rec_vals):.4f} +/- {np.std(rec_vals):.4f}",
            "Epsilon_Display": f"{np.mean(eps_list):.2f}" if cfg["enable_dp"] else "N/A (Non-DP)",
            "False_Alarm_Rate": f"{np.mean(fpr_vals):.4f} +/- {np.std(fpr_vals):.4f}"
        })
        
    df_dp = pd.DataFrame(summary_records)
    df_dp.to_csv(EXPERIMENTS_DIR / "dp_privacy_utility_tradeoff.csv", index=False)
    
    # Generate DP Tradeoff Curve Chart
    import matplotlib.pyplot as plt
    plt.style.use("tableau-colorblind10")
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})
    
    iid_dp = df_dp[df_dp["Partition"] == "iid"]
    plt.figure(figsize=(8, 4.5))
    x_eps = [15.0 if e is None else e for e in iid_dp["Epsilon_Mean"]]
    # Non-DP can be placed at right
    plt.plot(x_eps, iid_dp["F1_Mean"], marker="o", linewidth=2.2, color="#1f77b4", label="F1-Score")
    plt.plot(x_eps, iid_dp["Recall_Mean"], marker="s", linewidth=2.2, color="#2ca02c", label="Recall")
    for x, y, name in zip(x_eps, iid_dp["F1_Mean"], iid_dp["Configuration"]):
        plt.annotate(f"F1={y:.3f}", (x, y), textcoords="offset points", xytext=(0,10), ha="center", fontsize=9)
    plt.title("Differential Privacy Trade-off: Epsilon vs. F1 & Recall (IID, delta=1e-5)", fontsize=11, weight="bold", pad=10)
    plt.xlabel("Privacy Budget Epsilon (Lower = Stricter Privacy)", fontsize=11)
    plt.ylabel("Score", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    chart_p = CHARTS_DIR / "dp_epsilon_vs_f1.png"
    plt.savefig(chart_p, dpi=300)
    plt.close()
    
    print("\n" + "="*115, flush=True)
    print("           DIFFERENTIAL PRIVACY PRIVACY-UTILITY TRADEOFF BENCHMARK (5 SEEDS)           ", flush=True)
    print("="*115, flush=True)
    print(df_dp[["Configuration", "Epsilon_Display", "F1", "Recall", "False_Alarm_Rate", "DoS_Recall_Mean", "Web_Attack_Recall_Mean"]].to_string(index=False), flush=True)
    print("="*115 + "\n", flush=True)

if __name__ == "__main__":
    main()
