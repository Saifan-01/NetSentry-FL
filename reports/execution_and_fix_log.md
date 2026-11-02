# NetSentry-FL: Comprehensive Engineering & Research Execution Log
*A Complete Record of Foundations, Fixes, Experiments, and Verification for Month 5*

---

## 1. Foundation Fixes & Dataset Hygiene
### 1.1 Transition to Genuine Real Network Traffic
- **Source:** Canadian Institute for Cybersecurity (CIC-IDS2017) benchmark.
- **Traffic Components:**
  - `Wednesday-workingHours.pcap_ISCX.csv.parquet`: Real Denial of Service (DoS Hulk, DoS GoldenEye, DoS slowloris)
  - `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv.parquet`: Real reconnaissance PortScan
  - `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv.parquet`: Real Web Attacks (Brute Force, XSS)
  - Multi-day Benign baseline background traffic
- **Total Initial Records Combined:** 38,180 network flows.

### 1.2 Strict Preprocessing Pipeline (`src/data/preprocess.py`)
- **Deduplication:** Dropped **1,692 duplicate flows (4.43%)** *before* splitting to eliminate data snooping and artificial optimistic inflation.
- **Handling Infinities & Missing Values:** Dropped 28 invalid rows containing division-by-zero errors in flow rate calculations (`Flow Bytes/s`, `Flow Packets/s`).
- **Deterministic Column Pruning:** Replaced nondeterministic set dropping with `sorted(set(...))` and removed silent `errors="ignore"`. Pruned 10 invariant/zero-variance columns:
  `['Bwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd PSH Flags', 'Bwd URG Flags', ...]`
- **Active Feature Space:** Retained **68 clean numeric predictive flow features**.

### 1.3 Stratified 70 / 10 / 20 Split with Leak-Free Scaling
- **Partition Proportions:**
  - **Training Partition (70%):** 25,522 flows (16,553 Benign, 8,969 Attacks)
  - **Validation Partition (10%):** 3,646 flows (2,365 Benign, 1,281 Attacks)
  - **Held-Out Test Partition (20%):** 7,292 flows (4,729 Benign, 2,563 Attacks: 1,160 PortScan, 974 DoS, 429 Web Attacks)
- **Zero Leakage Guarantee:**
  - Scaler fitted **strictly on the 70% training partition (`X_train`)**, then applied to validation and test partitions without refitting.
  - Zero index or row overlap between train, val, and test splits (verified via automated unit test `tests/test_preprocessing.py`).

---

## 2. Root Cause Analysis: Fixing the Weak Centralized MLP Baseline

### 2.1 The Problem
In early Month 4 evaluation, Random Forest achieved an F1 of **0.9973**, while the Centralized PyTorch MLP achieved only **0.4299 F1** (recall: 27.70%). Because the MLP serves as the architectural foundation for federated training, this discrepancy undermined all federated comparisons.

### 2.2 Root Cause Discovery
Inspection of the scaled feature tensors revealed that raw network flow metrics in CIC-IDS2017 follow extreme heavy-tailed Pareto distributions (e.g., flow durations exceeding $10^8$ microseconds, packet byte rates exceeding $10^7$ bytes/sec). 
`RobustScaler` (which scales by median and interquartile range) left outlier values exceeding $1.2 \times 10^8$. When fed into PyTorch `BatchNorm1d` and `Linear` layers, these extreme inputs generated colossal activations, causing saturated gradients and neuron death in the hidden layers.

### 2.3 The Engineering Fix
1. Replaced `RobustScaler` with `QuantileTransformer(output_distribution="normal", n_quantiles=1000)` fitted **strictly on `X_train`**.
2. Mapped features to a Gaussian distribution and clipped scaled values strictly to $[-5.0, 5.0]$.
3. Result: Centralized MLP skyrocketed from **0.4299 to 0.9873 F1** (Accuracy: 99.09%, Precision: 97.63%, Recall: 99.84%, FPR: 1.31%).

### 2.4 Overfit Sanity Verification
A 200-sample training subset was fitted with AdamW for 100 epochs. The MLP reached **100.00% accuracy** with loss 0.00091, proving the model architecture is free of gradient bugs or label misalignment.

---

## 3. Privacy Evaluation: Centralized vs. Decentralized Feature Scaling

To evaluate the privacy implications of fitting scalers on pooled data in cross-silo FL, we developed an empirical comparison harness (`src/federated/scaling_privacy_analysis.py`):

| Scaling Paradigm | Privacy Level | F1-Score | Precision | Recall | False Alarm Rate (FPR) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Pooled QuantileTransformer** | Low (Assumes pooled training data) | **0.8498** | 99.12% | 74.37% | **0.36%** |
| **Local QuantileTransformer** | High (Zero raw data or quantile pooling) | **0.7900** | 98.77% | 65.82% | **0.44%** |
| **Fixed Parameterless Normalization** ($\log(1+x)$ + RFC bounds) | Complete (Zero fitting, 100% static) | **0.6963** | 98.78% | 53.77% | **0.36%** |

**Conclusion:** Local quantile fitting without data exchange incurs only a ~6% F1 penalty from client feature misalignment while preserving high precision (98.77%) and low false alarms (0.44%).

---

## 4. Month 5 Core Federated Experiments: 3-Seed Evaluation

Conducted across seeds 42, 101, and 777 on 7,292 held-out flows over 15 rounds:

### 4.1 Comparison Table (Mean $\pm$ Standard Deviation)

| Setup | Precision | Recall | F1-Score | PR-AUC | False Alarm Rate (FPR) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Centralized (MLP)** | 0.9812 $\pm$ 0.0020 | 0.9983 $\pm$ 0.0005 | **0.9897 $\pm$ 0.0008** | 0.9993 $\pm$ 0.0001 | 0.0104 $\pm$ 0.0011 (1.04%) |
| **Local-only (Client A - DoS)** | 0.9527 $\pm$ 0.0155 | 0.3835 $\pm$ 0.0053 | 0.5468 $\pm$ 0.0029 | 0.7841 $\pm$ 0.0386 | 0.0104 $\pm$ 0.0036 (1.04%) |
| **Local-only (Client B - PortScan)** | 0.9983 $\pm$ 0.0000 | 0.4521 $\pm$ 0.0002 | 0.6223 $\pm$ 0.0002 | 0.7786 $\pm$ 0.0060 | **0.0004 $\pm$ 0.0000 (0.04%)** |
| **Local-only (Client C - Web Attack)** | 0.8732 $\pm$ 0.0489 | 0.1900 $\pm$ 0.0086 | 0.3116 $\pm$ 0.0083 | 0.5640 $\pm$ 0.0270 | 0.0156 $\pm$ 0.0078 (1.56%) |
| **Federated, IID (FedAvg)** | 0.9702 $\pm$ 0.0013 | **0.9990 $\pm$ 0.0009** | **0.9844 $\pm$ 0.0009** | 0.9994 $\pm$ 0.0001 | 0.0166 $\pm$ 0.0007 (1.66%) |
| **Federated, non-IID Silos (FedAvg)** | 0.9895 $\pm$ 0.0010 | 0.7390 $\pm$ 0.0804 | 0.8437 $\pm$ 0.0514 | 0.9911 $\pm$ 0.0026 | 0.0042 $\pm$ 0.0003 (0.42%) |
| **Federated, non-IID Silos (FedProx $\mu=0.01$)** | 0.9897 $\pm$ 0.0028 | **0.8865 $\pm$ 0.0111** | **0.9352 $\pm$ 0.0059** | 0.9959 $\pm$ 0.0010 | 0.0050 $\pm$ 0.0014 (0.50%) |
| **Federated, non-IID Silos (FedProx $\mu=0.1$)** | 0.9887 $\pm$ 0.0050 | 0.6845 $\pm$ 0.0320 | 0.8086 $\pm$ 0.0238 | 0.9916 $\pm$ 0.0011 | 0.0042 $\pm$ 0.0017 (0.42%) |
| **Federated, non-IID Dir(0.5) (FedAvg)** | 0.9840 $\pm$ 0.0044 | 0.9662 $\pm$ 0.0398 | 0.9745 $\pm$ 0.0185 | 0.9973 $\pm$ 0.0023 | 0.0086 $\pm$ 0.0026 (0.86%) |
| **Federated, non-IID Dir(0.1) (FedAvg)** | 0.9863 $\pm$ 0.0116 | 0.8597 $\pm$ 0.0954 | 0.9152 $\pm$ 0.0505 | 0.9888 $\pm$ 0.0020 | 0.0071 $\pm$ 0.0064 (0.71%) |
| **Federated, non-IID Dir(0.1) (FedProx $\mu=0.01$)** | **0.9943 $\pm$ 0.0040** | 0.8559 $\pm$ 0.1003 | 0.9166 $\pm$ 0.0577 | 0.9894 $\pm$ 0.0056 | **0.0028 $\pm$ 0.0019 (0.28%)** |
| **Federated, non-IID Dir(0.1) (FedProx $\mu=0.1$)** | 0.9831 $\pm$ 0.0093 | 0.8970 $\pm$ 0.0517 | **0.9370 $\pm$ 0.0233** | 0.9878 $\pm$ 0.0051 | 0.0087 $\pm$ 0.0054 (0.87%) |

### 4.2 Key Discoveries
1. **Per-Attack Blindspot Proof:** Client B (trained on PortScan only) achieves **0.00% recall on DoS and Web Attacks**. Client A (DoS only) achieves **1.18% recall on PortScan**. The federated model catches all three families simultaneously.
2. **FedProx Prevents Minority Class Erasure:** On Extreme Silos, FedAvg achieves only 35.82% recall on Web Attacks with huge instability ($\pm 0.4576$). FedProx ($\mu=0.01$) boosts Web Attack recall to **87.65% with stable $\pm 0.0207$ variance**, lifting overall Silo F1 by **+9.15%**.
3. **Low False Alarms Across the Board:** All federated models maintain false alarm rates below 1.7%, reaching 0.28% on Dirichlet $\alpha=0.1$.

---

## 5. Artifacts and Demonstration Status
- **Interactive Streamlit Dashboard (`app/app.py`):** Fully operational, zero emojis, verified with strict schema alignment.
- **Publication Charts (`reports/charts/`):** 5 high-resolution figures generated at 300 DPI.
- **Git Synchronization:** All files committed and pushed to `https://github.com/Saifan-01/NetSentry-FL.git`.
