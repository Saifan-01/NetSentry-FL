"""
NetSentry-FL: Generate Publication-Quality Evaluation Charts
Plots:
1. Federated Convergence: F1-score across 15 communication rounds (IID vs Non-IID)
2. Per-Attack-Type Recall: Local Blindspots vs. Federated Detection (DoS, PortScan, Web Attack)
3. Comparative Metric Summary: Precision, Recall, F1 across Setups
4. Communication Cost Extension: Cumulative Network Overhead (KB) and Rounds to Target F1
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
CHARTS_DIR = PROJECT_ROOT / "reports" / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("tableau-colorblind10")
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})

def generate_all_charts():
    # 1. Convergence Chart (IID vs Non-IID F1 per round across seeds)
    conv_df = pd.read_csv(EXPERIMENTS_DIR / "convergence_history.csv")
    f1_col = "test_f1" if "test_f1" in conv_df.columns else "f1"
    plt.figure(figsize=(9, 5))
    
    for setup, grp in conv_df.groupby("setup"):
        avg_rnd = grp.groupby("round")[f1_col].mean()
        std_rnd = grp.groupby("round")[f1_col].std().fillna(0)
        line = plt.plot(avg_rnd.index, avg_rnd.values, marker="o", linewidth=2.0, label=setup)
        plt.fill_between(
            avg_rnd.index,
            avg_rnd.values - std_rnd.values,
            avg_rnd.values + std_rnd.values,
            alpha=0.15,
            color=line[0].get_color()
        )
        
    plt.title("Federated Convergence: Global Model F1-Score per Round (Mean +/- Std)", fontsize=12, weight="bold", pad=10)
    plt.xlabel("Communication Round", fontsize=11)
    plt.ylabel("Test F1-Score", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Federation Setup", frameon=True, fontsize=8.5, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "convergence_iid_vs_non_iid.png", dpi=300)
    plt.close()

    # 2. Per-Attack-Type Recall
    att_df = pd.read_csv(EXPERIMENTS_DIR / "per_attack_recall.csv")
    plt.figure(figsize=(9.5, 5))
    attacks = att_df["Attack"].unique()
    models = att_df["Model"].unique()
    x = np.arange(len(attacks))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        sub = att_df[att_df["Model"] == model]
        y_vals = [sub[sub["Attack"] == a]["Recall"].values[0] if len(sub[sub["Attack"] == a]) > 0 else 0.0 for a in attacks]
        plt.bar(x + i * width - 0.4 + width/2, y_vals, width, label=model)

    plt.xticks(x, attacks, fontsize=11, weight="bold")
    plt.title("Per-Attack-Type Recall: Local Silo Blindspots vs. Centralized/Federated Detection", fontsize=12, weight="bold", pad=10)
    plt.xlabel("Real CIC-IDS2017 Attack Category", fontsize=11)
    plt.ylabel("Detection Recall (TPR)", fontsize=11)
    plt.ylim(0, 1.15)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "per_attack_recall.png", dpi=300)
    plt.close()

    # 3. Setup Comparison (Precision, Recall, F1 across Setups)
    res_df = pd.read_csv(EXPERIMENTS_DIR / "rigorous_comparison_table.csv")
    plt.figure(figsize=(10, 5))
    x = np.arange(len(res_df))
    width = 0.25

    plt.bar(x - width, res_df["Precision_Mean"], width, yerr=res_df["Precision_Std"], capsize=3, label="Precision", color="#2ca02c")
    plt.bar(x, res_df["Recall_Mean"], width, yerr=res_df["Recall_Std"], capsize=3, label="Recall", color="#1f77b4")
    plt.bar(x + width, res_df["F1_Mean"], width, yerr=res_df["F1_Std"], capsize=3, label="F1-Score", color="#d62728")

    plt.xticks(x, res_df["Setup"], rotation=22, ha="right", fontsize=9.5)
    plt.ylabel("Score", fontsize=11)
    plt.title("Real CIC-IDS2017 Benchmark: Centralized vs. Local vs. Federated (Mean +/- Std)", fontsize=12, weight="bold", pad=10)
    plt.ylim(0, 1.15)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "setup_comparison_metrics.png", dpi=300)
    plt.close()

    # 4. Communication Overhead Chart
    comm_df = pd.read_csv(EXPERIMENTS_DIR / "communication_overhead.csv")
    plt.figure(figsize=(8, 4.2))
    plt.plot(comm_df["Round"], comm_df["Total_Network_Traffic_KB"], marker="s", color="#9467bd", linewidth=2.2, label="Cumulative Network Traffic (KB)")
    plt.title("Communication Overhead across Federated Rounds (3 Clients, 76.75 KB Model)", fontsize=12, weight="bold", pad=10)
    plt.xlabel("Communication Round", fontsize=11)
    plt.ylabel("Total Bandwidth Exchanged (KB)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "communication_cost.png", dpi=300)
    plt.close()

    print(f"[+] Successfully generated and saved all 4 publication charts to {CHARTS_DIR}")

if __name__ == "__main__":
    generate_all_charts()
