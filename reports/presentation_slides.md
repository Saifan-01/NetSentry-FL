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

*Key finding:* FedProx delivers a **+5.7 points** gain in F1-score and **+8.7 points** gain in Recall on Extreme Silos over FedAvg, winning in **4 out of 5 seeds**. Over the last 3 rounds (rounds 13–15), FedProx stabilizes to **0.9112 $\pm$ 0.0355** vs FedAvg's **0.8142 $\pm$ 0.0622**. On Dirichlet skew ($\alpha=0.1$), performance ranges overlap within standard error (no statistically clear difference).

---

### Slide 7: Dissecting the Web Attack Collapse & Minority Class Ablation
- **The Empirical Problem:**
  - Client C holds 1,500 attack flows out of 8,969 total attack flows in the training partition, representing **16.7% of all attacks** (while its 7,017 total flows represent 27.5% of overall training traffic).
  - In standard sample-weighted FedAvg ($w_k = n_k/N$), majority client updates overwrite Client C's gradients.
  - Result: Web Attack recall collapses to **33.47% $\pm$ 23.21%** under standard FedAvg.
- **Ablation Findings:**
  - **Equal Weighting ($w_k = 1/K = 1/3$):** Boosts Web Attack recall to **48.39% $\pm$ 36.08% (+14.9 points)** by mitigating parameter dilution.
  - **FedProx ($\mu^*=1.0$):** Restricts local parameter divergence, elevating mean Web Attack recall to **50.77% $\pm$ 46.35% (+17.3 points)** and DoS recall to **66.86% $\pm$ 16.07%**.
  - **Honest Statistical Caveat:** While FedProx improves the mean across 4 of 5 seeds, the minority class spread remains large ($\pm 46.35\%$), indicating sensitivity to weight initialization.

---

### Slide 8: Privacy-Preserving Feature Scaling Analysis (5 Seeds)
- **The Challenge:** Fitting global scalers assumes centralized pooling of private training flows.
- **Evaluation of Decentralized Normalization Alternatives (Extreme Silos, MLP, 5 Seeds):**
  - **Pooled QuantileTransformer:** F1: **0.8521 $\pm$ 0.0608**, Recall: **75.18% $\pm$ 9.22%**, FPR: **0.33% $\pm$ 0.17%**.
  - **Local QuantileTransformer (Zero Pooling):** F1: **0.8425 $\pm$ 0.0321**, Recall: **73.80% $\pm$ 4.79%**, FPR: **0.66% $\pm$ 0.14%** (incurs only a **1.0 point drop in F1** and **1.4 points drop in Recall**, while cutting variance in half).
  - **Fixed Parameterless Normalization ($\log(1+x)$ + RFC bounds):** F1: **0.8602 $\pm$ 0.0654**, Recall: **76.51% $\pm$ 9.60%**, FPR: **0.36% $\pm$ 0.10%**.
- **Takeaway:** Decentralized local quantile transformation and static parameterless normalization are viable, privacy-preserving drop-in replacements for pooled scalers in production cross-silo FL.

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

### Slide 10: Model Quantization & Communication Efficiency (5 Seeds)
- **Motivation:** Slashing network transmission payload in edge and multi-tenant security operations.
- **Evaluation Across 5 Seeds:**
  - **FP32 (Baseline):** 85.65 KB | F1: 0.9898 $\pm$ 0.0004 | Recall: 0.9980 $\pm$ 0.0012 | FPR: 1.01%
  - **FP16 (Half Precision):** 46.40 KB (**45.8% savings**) | F1: 0.9898 $\pm$ 0.0004 | Recall: 0.9980 $\pm$ 0.0012 | FPR: 1.01%
  - **INT8 (Dynamic Quantization):** **32.88 KB** (**61.6% savings**) | **F1: 0.9900 $\pm$ 0.0004** | **Recall: 0.9979 $\pm$ 0.0010** | **FPR: 0.98%**
- **Conclusion:** 61.6% bandwidth reduction with zero degradation in detection utility or false alarms.

---

### Slide 11: Byzantine Robustness: Defending Against Label-Flipping Attacks
- **Threat Model:** 1 malicious compromised client performing continuous label-flipping ($y \leftarrow 1 - y$).
- **Vulnerability of FedAvg:**
  - 3 Clients (33.3% malicious): FedAvg FPR surges to **2.36% $\pm$ 0.70%** (peak 3.34%), F1 drops from 0.9888 to 0.9738.
  - 5 Clients (20.0% malicious): FedAvg FPR nearly doubles to **2.11% $\pm$ 0.51%** (peak 2.96%), F1 drops to 0.9793.
- **Robust Aggregation Defenses (Coordinate Median & Trimmed Mean):**
  - **Coordinate Median (3 Clients):** Restores F1 to **0.9860 $\pm$ 0.0021** and suppresses FPR to **1.21% $\pm$ 0.23%**.
  - **Coordinate Median (5 Clients):** Restores F1 to **0.9870 $\pm$ 0.0011** and suppresses FPR to **1.26% $\pm$ 0.18%** (matching the 1.25% benign baseline).
  - **Trimmed Mean:** Yields identical robust recovery, filtering malicious gradient updates.

---

### Slide 12: Formal Differential Privacy via Opacus (DP-SGD)
- **Implementation:** Per-sample gradient clipping ($C=1.0$) and calibrated Gaussian noise injection with Opacus PrivacyEngine.
- **Architecture:** GroupNorm MLP (`norm_type="group_norm"`), guaranteeing batch-size independence and Opacus compliance. Fair comparison using identical GroupNorm for Non-DP baseline.
- **Privacy-Utility Tradeoff ($\delta=10^{-5}$, 5 Seeds):**
  - **Non-DP Baseline:** F1: 0.9851 $\pm$ 0.0032 | Recall: 99.94% | $\epsilon = \infty$
  - **DP-SGD ($\sigma=0.5$):** F1: 0.9548 $\pm$ 0.0013 | Recall: 93.22% | $\epsilon = 26.40$
  - **DP-SGD ($\sigma=1.0$):** **F1: 0.9486 $\pm$ 0.0020** | **Recall: 92.20%** | $\mathbf{\epsilon = 4.06}$ (strong mathematical guarantee with only 3.6 points F1 cost)
  - **DP-SGD ($\sigma=2.0$):** F1: 0.9340 $\pm$ 0.0064 | Recall: 89.61% | $\mathbf{\epsilon = 1.41}$ (extreme privacy budget, $\epsilon < 1.5$)
- **DP on Extreme Silos (FedProx $\mu=1.0$, $\sigma=1.0$):**
  - F1: **0.9384 $\pm$ 0.0015** | $\epsilon = 4.10$ | Web Attack Recall: **84.62%**.
  - Per-sample gradient clipping prevents gradient explosion from majority DoS clients, stabilizing Silo training.

---

### Slide 13: Summary of Months 1 to 6 & Roadmap to Month 7
1. **Months 1–3:** Prototype baseline pipeline, 3-client FedAvg prototype.
2. **Month 4:** Transition to real CIC-IDS2017 benchmark, deterministic preprocessing, fixed MLP baseline (F1 0.43 -> 0.989).
3. **Month 5:** Core heterogeneity experiments, per-attack blindspots, FedProx stabilization, decentralized scaling privacy analysis.
4. **Month 6:** Model quantization (61.6% compression), Byzantine defense against label poisoning, Opacus DP-SGD ($\epsilon=4.06$).
5. **Month 7 Roadmap (Packaging & Final Defense):**
   - End-to-end containerized client-server deployment (Docker/Compose).
   - Automated regression testing and continuous evaluation.
   - Comprehensive final defense report and interactive executive dashboard.
