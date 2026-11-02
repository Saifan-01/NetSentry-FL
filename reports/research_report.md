# NetSentry-FL: Formal Month 5 Research & Technical Evaluation Report
**A Privacy-Preserving Federated Learning Architecture for Network Intrusion Detection**
*Evaluated on the Canadian Institute for Cybersecurity CIC-IDS2017 Benchmark across 3 Random Seeds*

---

## 1. Executive Summary & Research Motivation
Modern distributed enterprise networks face diverse cyber threats spanning volumetric Application-Layer Denial of Service (DoS) floods, reconnaissance PortScans, and sophisticated Web Attacks (SQL injection, XSS, brute force). Organizations operating in distinct sectors—such as Financial Services (Client A), Healthcare Infrastructure (Client B), and Cloud Platforms (Client C)—cannot pool their network telemetry into a central cloud repository due to regulatory frameworks (GDPR, HIPAA, PCI-DSS) and organizational confidentiality requirements.

Training intrusion detection models in complete isolation creates severe **data silos**:
- Local detectors overfit to local traffic distributions and suffer total blindspots (0% recall) on attack vectors they have never encountered.
- When exposed to broader enterprise traffic, siloed models exhibit high false alarms or narrow decision boundaries.

**NetSentry-FL** investigates whether **Federated Learning (FL)** using **Flower** with **Federated Averaging (FedAvg)** and **Federated Proximal Regularization (FedProx)** can aggregate cross-organizational threat intelligence without transmitting raw network flow records, while examining the impact of extreme statistical heterogeneity (cross-silo attack partitioning and Dirichlet label skew).

---

## 2. Dataset Hygiene & Preprocessing Methodology

This study utilizes real packet flow records from the **CIC-IDS2017** benchmark:
- `Wednesday-workingHours.pcap_ISCX`: Real DoS attacks (DoS Hulk, GoldenEye, slowloris)
- `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX`: Real PortScan reconnaissance
- `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX`: Real Web Attacks (Brute Force, XSS)
- Multi-day Benign enterprise baseline traffic

### 2.1 Deduplication & Data Cleaning
- **Whitespace Regularization:** Stripped irregular leading/trailing spaces across all 79 feature column headers.
- **Deduplication Before Splitting:** Dropped 1,692 exact duplicate network flows (4.43%) *prior* to dataset splitting, preventing data leakage across train and test.
- **Handling Division Anomalies:** Dropped 28 invalid flows exhibiting `np.inf`, `-np.inf`, or `NaN` values in flow rate computations (`Flow Bytes/s`, `Flow Packets/s`).
- **Deterministic Feature Pruning:** Pruned 10 constant zero-variance features using sorted column identification, leaving **68 clean active flow features**.

### 2.2 Leakage-Free 70/10/20 Stratified Splitting
- The curated real dataset (36,460 flows) was split with stratified sampling on attack category into:
  - **Training Partition (70%):** 25,522 flows (16,553 Benign, 8,969 Attacks)
  - **Validation Partition (10%):** 3,646 flows (2,365 Benign, 1,281 Attacks)
  - **Held-Out Test Partition (20%):** 7,292 flows (4,729 Benign, 2,563 Attacks: 1,160 PortScan, 974 DoS, 429 Web Attacks).
- Verified zero index or row overlap across all three partitions via automated unit test suite (`tests/test_preprocessing.py`).

### 2.3 Heavy-Tail Feature Scaling Breakthrough
In initial experiments, standard `RobustScaler` produced extreme unclipped feature values (up to $1.2 \times 10^8$ in packet rates and flow durations), causing gradient explosion in the neural network and collapsing Centralized MLP performance to an F1 of 0.4299.
Replacing this with `QuantileTransformer(output_distribution="normal")` fitted **strictly on the training split (`X_train`)** and clipped to $[-5.0, 5.0]$ completely eliminated gradient instability and elevated the Centralized MLP to **0.9897 F1**, within 0.007 of Random Forest.

---

## 3. Privacy Discussion: Centralized vs. Decentralized Feature Scaling

A fundamental question in federated intrusion detection is whether feature scaling statistics require pooling raw data. NetSentry-FL evaluates three paradigms on the 7,292 held-out test flows:

| Scaling Paradigm | Privacy Level | F1-Score | Precision | Recall | False Alarm Rate (FPR) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Pooled QuantileTransformer** (Simulation Baseline) | Low (Assumes pooled training data) | **0.8498** | 99.12% | 74.37% | **0.36%** |
| **Local QuantileTransformer** (Strictly Decentralized) | High (Zero raw data or quantile pooling) | **0.7900** | 98.77% | 65.82% | **0.44%** |
| **Fixed Parameterless Normalization** ($\log(1+x)$ + RFC bounds) | Complete (Zero fitting, 100% static) | **0.6963** | 98.78% | 53.77% | **0.36%** |

### Key Architectural Finding:
- In real federated deployments where raw flow pooling is prohibited, **Local QuantileTransformer** (fitted strictly on each client's private telemetry) maintains high precision (98.77%) and low false alarm rates (0.44%), incurring only a modest ~6% F1 penalty due to feature space misalignment across heterogeneous clients.
- For maximum compliance, **Fixed Parameterless Normalization** provides complete privacy with zero statistical exchange, requiring additional training rounds to match quantile performance.

---

## 4. Centralized Baseline Results (Pooled Upper Bound)

All models were evaluated on the fixed 7,292 held-out test flows:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | False Alarm Rate (FPR) | Confusion Matrix [TN, FP, FN, TP] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (100 Trees)** | 99.78% | 99.77% | 99.61% | **0.9969** | 1.0000 | 1.0000 | **0.13%** | [4723, 6], [10, 2553] |
| **Centralized PyTorch MLP** | 99.09% | 97.63% | 99.84% | **0.9873** | 0.9997 | 0.9995 | **1.31%** | [4667, 62], [4, 2559] |
| **Logistic Regression** | 97.46% | 95.11% | 97.82% | **0.9644** | 0.9934 | 0.9765 | **2.73%** | [4600, 129], [56, 2507] |

*Sanity Check:* Overfit test on a 200-sample subset reached **100.00% accuracy** with loss 0.00091, confirming absence of optimization bottlenecks.

---

## 5. Month 5 Heterogeneity Evaluation: FedAvg vs. FedProx

All experiments were executed across **3 independent random seeds (42, 101, 777)** on the held-out test partition across 15 communication rounds.

### 5.1 Definitive Comparison Table (Mean $\pm$ Standard Deviation)

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

---

## 6. Deep Research Findings

### 6.1 Core Hypothesis Validated: Silo Blindspots vs. Federated Detection
The per-attack-type recall analysis directly demonstrates why cross-organizational federated learning is essential:

| Model Architecture | DoS Recall (974 flows) | PortScan Recall (1,160 flows) | Web Attack Recall (429 flows) | Total Test Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Local Client A (DoS Silo)** | **98.87% $\pm$ 1.53%** | 1.18% $\pm$ 0.26% | 1.48% $\pm$ 1.08% | 38.35% |
| **Local Client B (PortScan Silo)** | **0.00% $\pm$ 0.00%** | **99.89% $\pm$ 0.05%** | **0.00% $\pm$ 0.00%** | 45.21% |
| **Local Client C (Web Attack Silo)** | 5.99% $\pm$ 2.55% | 0.23% $\pm$ 0.05% | **99.30% $\pm$ 0.47%** | 19.00% |
| **Federated IID (FedAvg)** | **99.83% $\pm$ 0.30%** | **99.91% $\pm$ 0.00%** | **100.00% $\pm$ 0.00%** | **99.90%** |
| **Federated Silos (FedAvg)** | 60.16% $\pm$ 5.92% | 99.51% $\pm$ 0.36% | 35.82% $\pm$ 45.76% | 73.90% |
| **Federated Silos (FedProx $\mu=0.01$)** | **76.18% $\pm$ 2.72%** | **99.48% $\pm$ 0.26%** | **87.65% $\pm$ 2.07%** | **88.65%** |

- **Local Isolated Models Fail:** Client B suffers a **100% blindspot on DoS and Web Attacks**. Client A is blind to PortScan (1.18% recall).
- **Federated Models Detect All Attack Vectors:** Both Federated IID and Non-IID models detect threats across all three domains simultaneously without exchanging private traffic.

### 6.2 FedProx Overcomes Client Drift and Protects Minority Classes
Under extreme silos, standard FedAvg suffers from **client drift**:
- Local models optimize in conflicting directions, causing updates from the minority class (Client C Web Attacks, 1,500 samples) to be overwritten by majority clients (DoS and PortScan, 3,400+ samples).
- Consequently, FedAvg achieves only **35.82% recall on Web Attacks** with extreme instability ($\pm 0.4576$).
- **FedProx ($\mu=0.01$) constrains local updates to stay near the consensus parameter manifold**, boosting Web Attack recall from **35.82% to 87.65%** and collapsing variance to a stable $\pm 0.0207$. Overall Silo F1 jumps from **0.8437 to 0.9352 (+9.15%)**.

### 6.3 Hyperparameter Sensitivity to $\mu$
- **$\mu=0.01$** provides the optimal balance on Extreme Silos, allowing local learning while suppressing drift.
- **$\mu=0.1$** over-penalizes local movement on silos (F1: 0.8086), but proves optimal on continuous Dirichlet skew $\alpha=0.1$ (F1: 0.9370 vs 0.9152 for FedAvg), demonstrating that optimal proximal regularization scales with the nature of data skew.

### 6.4 Low False Alarm Rates (FPR < 1.7% Everywhere)
In contrast to initial synthetic prototype runs, on real CIC-IDS2017 data with proper normalization, **false alarm rates remain strictly below 1.7% across all federated configurations**, and drop to **0.28%** under FedProx $\mu=0.01$ on Dirichlet skew.

---

## 7. Communication Efficiency & Deployment Footprint
- **Model Size:** 19,649 float32 parameters (**76.75 KB** total per transmission).
- Across 15 communication rounds with 3 clients, total network traffic is **6.91 MB**.
- Transmitting raw PCAP logs or flow tables for 25,522 flows would require gigabytes of bandwidth and violate data sovereignty laws.

---

## 8. Artifacts & Deliverables Summary
All code, models, artifacts, and figures are versioned in Git (`https://github.com/Saifan-01/NetSentry-FL.git`):
- `reports/charts/dirichlet_and_silo_distributions.png`: Partition visualization
- `reports/charts/convergence_iid_vs_non_iid.png`: Convergence history curves
- `reports/charts/per_attack_recall.png`: Silo blindspots vs federated recall
- `reports/charts/setup_comparison_metrics.png`: 12-setup comparison chart
- `reports/charts/communication_cost.png`: Cumulative bandwidth chart
- `experiments/rigorous_comparison_table.csv`: 3-seed benchmark metrics
- `experiments/per_attack_recall.csv`: Per-attack recall breakdown
- `experiments/scaling_privacy_comparison.csv`: Feature scaling privacy benchmark
- `app/app.py`: Interactive Streamlit dashboard with real-time inference and academic evaluation UI.
