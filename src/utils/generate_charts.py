import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

EXPERIMENTS_DIR = Path("experiments")
CHARTS_DIR = Path("reports/charts")
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("tableau-colorblind10")

# 1. Convergence Chart (IID vs Non-IID F1 per round)
conv_df = pd.read_csv(EXPERIMENTS_DIR / "convergence_history.csv")
plt.figure(figsize=(9, 5))
for setup, grp in conv_df.groupby("setup"):
    avg_rnd = grp.groupby("round")["f1"].mean()
    plt.plot(avg_rnd.index, avg_rnd.values, marker="o", linewidth=2.5, label=setup)
plt.title("Federated Convergence: Global Model F1-Score per Communication Round", fontsize=13, weight="bold", pad=12)
plt.xlabel("Communication Round", fontsize=11)
plt.ylabel("Test F1-Score", fontsize=11)
plt.ylim(0.4, 1.05)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(title="Aggregation Setup", frameon=True)
plt.tight_layout()
plt.savefig(CHARTS_DIR / "convergence_iid_vs_non_iid.png", dpi=300)
plt.close()

# 2. Per-Attack-Type Recall (Highlighting Local Blindspots vs Federated)
att_df = pd.read_csv(EXPERIMENTS_DIR / "per_attack_recall.csv")
plt.figure(figsize=(10, 5.5))
attacks = att_df["Attack"].unique()
models = att_df["Model"].unique()
x = np.arange(len(attacks))
width = 0.8 / len(models)

for i, model in enumerate(models):
    sub = att_df[att_df["Model"] == model]
    y_vals = [sub[sub["Attack"] == a]["Recall"].values[0] if len(sub[sub["Attack"] == a]) > 0 else 0.0 for a in attacks]
    plt.bar(x + i * width - 0.4 + width/2, y_vals, width, label=model)

plt.xticks(x, attacks, fontsize=11)
plt.title("Per-Attack-Type Recall: Local Blindspots vs. Federated Generalization", fontsize=13, weight="bold", pad=12)
plt.xlabel("Network Attack Category", fontsize=11)
plt.ylabel("Detection Recall (TPR)", fontsize=11)
plt.ylim(0, 1.15)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
plt.tight_layout()
plt.savefig(CHARTS_DIR / "per_attack_recall.png", dpi=300)
plt.close()

# 3. Setup Comparison (F1, Recall, Precision across Setups)
res_df = pd.read_csv(EXPERIMENTS_DIR / "rigorous_comparison_table.csv")
plt.figure(figsize=(10, 5))
x = np.arange(len(res_df))
width = 0.25

plt.bar(x - width, res_df["Precision"], width, label="Precision", color="#2ca02c")
plt.bar(x, res_df["Recall"], width, label="Recall", color="#1f77b4")
plt.bar(x + width, res_df["F1"], width, label="F1-Score", color="#d62728")

plt.xticks(x, res_df["Setup"], rotation=20, ha="right", fontsize=10)
plt.ylabel("Score", fontsize=11)
plt.title("3-Seed Rigorous Benchmark: Centralized vs. Local vs. Federated", fontsize=13, weight="bold", pad=12)
plt.ylim(0, 1.15)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(CHARTS_DIR / "setup_comparison_metrics.png", dpi=300)
plt.close()

# 4. Communication Cost Chart
comm_df = pd.read_csv(EXPERIMENTS_DIR / "communication_overhead.csv")
plt.figure(figsize=(8, 4.5))
plt.plot(comm_df["Round"], comm_df["Total_Network_Traffic_KB"], marker="s", color="#9467bd", linewidth=2.5, label="Total Cumulative Network Traffic (KB)")
plt.title("Communication Overhead across Federated Rounds (3 Clients)", fontsize=13, weight="bold", pad=12)
plt.xlabel("Communication Round", fontsize=11)
plt.ylabel("Cumulative Exchange (KB)", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(CHARTS_DIR / "communication_cost.png", dpi=300)
plt.close()

print(f"[+] All 4 publication-quality charts saved to {CHARTS_DIR}")
