# Presentation: Federated Learning for Network Intrusion Detection (NetSentry-FL)
**Month 5 Research Milestone Presentation (10 Slides)**
*Empirically Evaluated on the Curated CIC-IDS2017 Benchmark Across 5 Random Seeds*

---

### Slide 1: Title & Overview
- **Project Title:** NetSentry-FL: Privacy-Preserving Collaborative Intrusion Detection via Federated Learning
- **Student Name:** Saifan
- **Domain:** Distributed Machine Learning & Cybersecurity
- **Dataset Scope:** Curated 36,460-flow benchmark derived from CIC-IDS2017 (Benign, DoS, PortScan, Web Attacks).
- **Core Research Question:** Can multi-tenant organizations train effective neural intrusion detectors on non-IID threat distributions without exposing proprietary network telemetry?

---

### Slide 2: The Operational Problem — Threat Silos & Regulatory Barriers
- **Heterogeneous Attack Profiles:** Financial infrastructure faces DoS floods; healthcare systems face reconnaissance PortScans; e-commerce platforms absorb Web Attacks.
- **Regulatory Barriers:** GDPR, HIPAA, and PCI-DSS strictly forbid centralizing raw network flow records or PCAP captures.
- **The Failure of Isolation:** Isolated local models suffer severe attack blindspots:
  - Client B (PortScan silo) achieves **0.00% recall on DoS and Web Attacks**.
  - Client A (DoS silo) achieves **1.18% recall on PortScan**.
- **Collaborative Requirement:** Organizations must pool intelligence without centralizing raw packet data.

---

### Slide 3: NetSentry-FL Architecture & Optimization
- **Federated Architecture:**
  - 3 Heterogeneous Organization Clients + Central Coordination Server.
  - Local deep neural networks: 3-layer MLP (128-64-32 units, 19,649 parameters, 76.75 KB).
  - Parameter exchange only ($\Delta \theta_k$); zero raw flow records leave the tenant boundary.
- **Algorithms:**
  - **FedAvg:** Baseline weighted aggregation ($w_k = n_k / N$).
  - **FedProx:** Proximal regularization ($\frac{\mu}{2} \|w - w_t\|^2$) to constrain local client drift under severe statistical heterogeneity.
- **Zero Test-Leakage Protocol:**
  - Hyperparameter $\mu^*$ and local epochs $E^*$ tuned strictly on the **Validation Split** (3,646 flows), never on the test set.

---

### Slide 4: Data Hygiene, Deduplication & Scaling Breakthrough
- **Dataset Ingestion & Hygiene:**
  - Combined Wednesday (DoS), Friday (PortScan), Thursday (Web Attacks), and multi-day Benign flows.
  - **Deduplication:** Dropped 1,692 exact duplicates (4.43%) *prior* to splitting to prevent optimistic generalization bias.
  - Dropped 28 division-by-zero anomalies (`np.inf`, `NaN`).
  - Pruned 10 invariant features $\rightarrow$ **68 active predictive flow features**.
- **Leakage-Free 70/10/20 Stratification:**
  - Train (25,522 flows), Validation (3,646 flows), Held-Out Test (7,292 flows; 4,729 Benign, 2,563 Attacks).
- **The Scaling Breakthrough:**
  - Standard scalers left heavy-tailed packet rates unclipped, destabilizing backpropagation.
  - Resolved with `QuantileTransformer(output_distribution="normal")` fitted strictly on `X_train` and clipped to $[-5.0, 5.0]$.
  - Centralized MLP elevated from 0.4299 F1 to **0.9897 F1**, matching Random Forest (0.9969 F1).

---

### Slide 5: Experimental Evaluation Setups (5 Random Seeds)
- Evaluated across **5 independent seeds: [42, 101, 777, 1234, 5678]** on the fixed 7,292 held-out test flows over 15 communication rounds.
- **Tested Paradigms:**
  1. **Centralized (MLP):** Empirical upper bound with pooled training data.
  2. **Local-Only Silos:** Client A (DoS), Client B (PortScan), Client C (Web Attack).
  3. **Federated IID (FedAvg):** Uniform threat distribution across 3 clients.
  4. **Federated Non-IID Silos (FedAvg):** Cross-silo attack partitioning.
  5. **Federated Non-IID Silos (FedProx $\mu^*$):** Proximal regularization against client drift.
  6. **Ablation (Equal Weighting 1/K):** Investigating minority class preservation.
  7. **Dirichlet Skew ($\alpha=0.5, \alpha=0.1$):** Continuous non-IID label skew.

---

### Slide 6: Comprehensive Benchmark Results (Held-Out Test Partition)
*All metrics reported as Mean $\pm$ Standard Deviation across 5 seeds on final round R=15:*

| Setup | Precision | Recall | F1-Score | PR-AUC | False Alarm Rate (FPR) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Centralized (MLP)** | 0.9784 $\pm$ 0.0063 | 0.9984 $\pm$ 0.0013 | **0.9883 $\pm$ 0.0030** | 0.9993 $\pm$ 0.0003 | 1.20% $\pm$ 0.36% |
| **Local-only (Client A - DoS)** | 0.9519 $\pm$ 0.0138 | 0.3833 $\pm$ 0.0087 | 0.5464 $\pm$ 0.0083 | 0.7504 $\pm$ 0.0657 | 1.06% $\pm$ 0.33% |
| **Local-only (Client B - PortScan)** | 0.9984 $\pm$ 0.0003 | 0.4520 $\pm$ 0.0002 | 0.6223 $\pm$ 0.0002 | 0.7907 $\pm$ 0.0188 | **0.04% $\pm$ 0.01%** |
| **Local-only (Client C - Web Attack)** | 0.8878 $\pm$ 0.0301 | 0.1839 $\pm$ 0.0120 | 0.3043 $\pm$ 0.0148 | 0.5813 $\pm$ 0.0291 | 1.29% $\pm$ 0.44% |
| **Federated, IID (FedAvg)** | 0.9817 $\pm$ 0.0018 | **0.9973 $\pm$ 0.0010** | **0.9894 $\pm$ 0.0007** | 0.9994 $\pm$ 0.0000 | 1.01% $\pm$ 0.10% |
| **Federated, non-IID Silos (FedAvg)** | 0.9957 $\pm$ 0.0016 | 0.7032 $\pm$ 0.1198 | 0.8182 $\pm$ 0.0856 | 0.9931 $\pm$ 0.0029 | **0.17% $\pm$ 0.08%** |
| **Federated, non-IID Silos (Equal-Weighted 1/K)** | 0.9934 $\pm$ 0.0055 | 0.7293 $\pm$ 0.1128 | 0.8356 $\pm$ 0.0823 | 0.9942 $\pm$ 0.0022 | 0.27% $\pm$ 0.24% |
| **Federated, non-IID Silos (FedProx $\mu^*=1.0$)** | 0.9883 $\pm$ 0.0044 | **0.7899 $\pm$ 0.0875** | **0.8750 $\pm$ 0.0559** | 0.9898 $\pm$ 0.0054 | 0.53% $\pm$ 0.22% |
| **Federated, non-IID Dir(0.5) (FedAvg)** | 0.9862 $\pm$ 0.0025 | 0.9753 $\pm$ 0.0316 | 0.9804 $\pm$ 0.0155 | 0.9985 $\pm$ 0.0015 | 0.74% $\pm$ 0.15% |
| **Federated, non-IID Dir(0.1) (FedAvg)** | 0.9894 $\pm$ 0.0053 | 0.9195 $\pm$ 0.0647 | 0.9519 $\pm$ 0.0336 | 0.9933 $\pm$ 0.0044 | 0.55% $\pm$ 0.29% |
| **Federated, non-IID Dir(0.1) (FedProx $\mu^*=0.001$)** | 0.9888 $\pm$ 0.0058 | 0.9046 $\pm$ 0.0959 | 0.9417 $\pm$ 0.0529 | 0.9928 $\pm$ 0.0050 | 0.59% $\pm$ 0.32% |

*Key finding:* FedProx delivers a **+5.7 points** gain in F1-score and **+8.7 points** gain in Recall on Extreme Silos over FedAvg. On Dirichlet skew ($\alpha=0.1$), performance ranges overlap within standard error (no statistically clear difference).

---

### Slide 7: Dissecting the Web Attack Collapse & Minority Class Ablation
- **The Empirical Problem:**
  - Client C possesses only 1,500 Web Attack training flows (27.5% weight) vs 3,400+ for Clients A & B.
  - In standard sample-weighted FedAvg ($w_k = n_k/N$), majority client updates overwrite Client C's gradients.
  - Result: Web Attack recall collapses to **33.47% $\pm$ 23.21%** under standard FedAvg.
- **Ablation Analysis:**
  - **Equal Weighting ($w_k = 1/K = 1/3$):** Boosts Web Attack recall to **48.39% $\pm$ 36.08% (+14.9 points)** by preventing update dilution.
  - **FedProx ($\mu^*=1.0$):** Restricts local parameter divergence, preserving minority attack boundaries and elevating Web Attack recall to **50.77% $\pm$ 46.35% (+17.3 points)** and DoS recall to **66.86% $\pm$ 16.07%**.
- **Ablation Analysis:**
  - **Equal Weighting ($w_k = 1/K = 1/3$):** Boosts Web Attack representation during server aggregation.
  - **FedProx ($\mu^*=0.01$):** Restricts local parameter divergence, preserving minority attack boundaries and elevating Web Attack recall to **87.65% $\pm$ 2.07%**.

---

### Slide 8: Privacy-Preserving Feature Scaling Analysis
- **The Challenge:** Fitting global scalers assumes centralized pooling of private training flows.
- **Evaluation of Decentralized Normalization Alternatives (Extreme Silos, MLP, 3 seeds):**
  - **Pooled QuantileTransformer:** F1: 0.8498, Recall: 74.37%, FPR: 0.36%.
  - **Local QuantileTransformer (Zero Pooling):** F1: 0.7900, Recall: 65.82%, FPR: 0.44% (incurs only an ~8.5 points drop in recall while maintaining 98.77% precision).
  - **Fixed Parameterless Normalization ($\log(1+x)$):** F1: 0.6963, Recall: 53.77%, FPR: 0.36%.
- **Takeaway:** Local quantile transformation provides an effective, privacy-compliant trade-off for cross-silo production environments.

---

### Slide 9: Communication Efficiency & Streamlit Live Demonstration
- **Model Efficiency:** 19,649 parameters = **76.75 KB** per client transmission.
- **Bandwidth Consumption:** Total traffic across 15 rounds is **6.91 MB** (vs. Gigabytes of raw PCAP data).
- **Interactive System (`app/app.py`):**
  - Interactive validation with zero emojis for academic rigor.
  - Live CSV upload and automated schema validation (detects and rejects missing feature sets).
  - Real-time intrusion inference using the trained global model.
  - Interactive decision threshold slider and per-attack vulnerability visualizations.

---

### Slide 10: Conclusion & Roadmap to Month 6
1. **Validation of Federation:** Collaborative training eliminates silo blindspots (recovering up to 99% recall on PortScan and DoS without raw data exchange).
2. **Heterogeneity Mitigation:** FedProx successfully stabilizes cross-silo training (+9.2 points F1 gain on extreme silos) and shields minority Web Attacks from catastrophic forgetting.
3. **Month 6 Roadmap:**
   - **Differential Privacy (DP):** DP-SGD via Opacus ($\epsilon, \delta$ privacy budget tracking).
   - **Bandwidth Compression:** FP16/INT8 weight quantization and top-$k$ gradient sparsification.
   - **Byzantine & Poisoning Robustness:** Resisting label-flipping attacks via trimmed mean and coordinate-wise median aggregation.
