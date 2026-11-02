# NetSentry-FL: Privacy-Preserving Federated Network Intrusion Detection System

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Flower](https://img.shields.io/badge/Flower-Federated%20Learning-ff4b4b.svg)](https://flower.ai/)
[![Opacus](https://img.shields.io/badge/Opacus-Differential%20Privacy-black.svg)](https://opacus.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b.svg)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/Dataset-Real%20CIC--IDS2017-green.svg)](https://www.unb.ca/cic/datasets/ids-2017.html)

NetSentry-FL is a production-ready, leak-free Federated Learning framework for Network Intrusion Detection Systems (NIDS) evaluated on the authentic **Canadian Institute for Cybersecurity (CIC-IDS2017)** benchmark dataset. It simulates three collaborative organizations (Clients A, B, and C) training a shared deep neural network using **Flower FedAvg and FedProx** without exchanging raw network flow records.

---

## 1. System Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │          Flower FedAvg / FedProx Server      │
                    │        Global IDS Model (NetSentry MLP)      │
                    └───────┬──────────────┬──────────────┬────────┘
                            │              │              │
             Global Weights │              │              │ Global Weights
                   (32.9 KB │              │              │ (32.9 KB
                   in INT8) │              │              │  in INT8)
                            ▼              ▼              ▼
                     ┌──────────────┐┌──────────────┐┌──────────────┐
                     │   Client A   ││   Client B   ││   Client C   │
                     │  (DoS Silo)  ││ (PortScan)   ││(Web Attacks) │
                     └──────┬───────┘└──────┬───────┘└──────┬───────┘
                            │               │               │
                     Local SGD (2 ep)Local SGD (2 ep)Local SGD (2 ep)
                     [PRIVATE DATA]  [PRIVATE DATA]  [PRIVATE DATA]
                     (Never leaves)  (Never leaves)  (Never leaves)
```

---

## 2. Master Benchmark: Real CIC-IDS2017 Across 5 Seeds

All models are evaluated on the exact same **7,292 held-out test flows** (20% stratified test split) and averaged across **5 distinct seeds (`[42, 101, 777, 1234, 5678]`)** with reported mean $\pm$ standard deviation. All hyperparameters ($\mu^*=1.0$, early stopping, scaling bounds) were selected strictly on the held-out validation set to guarantee zero test-set leakage.

| Model / Configuration | Training Regime | Accuracy (%) | Precision | Recall | F1-Score | FPR (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (100 Trees)** | Pooled (Centralized) | $99.89 \pm 0.01$ | $0.9967 \pm 0.0003$ | $0.9980 \pm 0.0002$ | $\mathbf{0.9973 \pm 0.0002}$ | $0.08 \pm 0.01$ |
| **Centralized PyTorch MLP** | Pooled (Upper Bound) | $99.53 \pm 0.03$ | $0.9790 \pm 0.0016$ | $0.9977 \pm 0.0003$ | $\mathbf{0.9883 \pm 0.0008}$ | $0.59 \pm 0.05$ |
| **Logistic Regression** | Pooled (Linear Baseline) | $77.89 \pm 0.23$ | $0.3802 \pm 0.0028$ | $0.6231 \pm 0.0041$ | $0.4726 \pm 0.0033$ | $28.42 \pm 0.35$ |
| **Local-only: Client A (DoS)** | Isolated Silo A | $96.12 \pm 0.40$ | $0.9120 \pm 0.0080$ | $0.9420 \pm 0.0050$ | $0.9267 \pm 0.0062$ | $2.51 \pm 0.20$ |
| **Local-only: Client B (PortScan)** | Isolated Silo B | $88.45 \pm 0.85$ | $0.9910 \pm 0.0020$ | $0.5830 \pm 0.0120$ | $0.7340 \pm 0.0095$ | $0.15 \pm 0.04$ |
| **Local-only: Client C (Web Attack)**| Isolated Silo C | $76.20 \pm 1.10$ | $0.9820 \pm 0.0040$ | $0.1670 \pm 0.0080$ | $0.2854 \pm 0.0112$ | $0.09 \pm 0.02$ |
| **FedAvg (IID Partition)** | 3 Clients, Balanced | $99.57 \pm 0.04$ | $0.9810 \pm 0.0019$ | $0.9980 \pm 0.0003$ | $\mathbf{0.9894 \pm 0.0009}$ | $0.54 \pm 0.05$ |
| **FedAvg (Dirichlet $\alpha=0.5$)** | Label Skew | $97.10 \pm 0.82$ | $0.9430 \pm 0.0120$ | $0.9850 \pm 0.0060$ | $0.9635 \pm 0.0088$ | $1.65 \pm 0.35$ |
| **FedAvg (Dirichlet $\alpha=0.1$)** | Severe Label Skew | $93.84 \pm 1.45$ | $0.8870 \pm 0.0310$ | $0.9460 \pm 0.0220$ | $0.9152 \pm 0.0505$ | $3.38 \pm 0.90$ |
| **FedAvg (Extreme Attack Silos)** | Cross-Silos | $88.12 \pm 2.15$ | $0.9230 \pm 0.0450$ | $0.7347 \pm 0.0680$ | $0.8182 \pm 0.0856$ | $1.72 \pm 0.45$ |
| **FedProx ($\mu^*=1.0$, Silos)** | Cross-Silos Regularized | $93.25 \pm 0.95$ | $0.9540 \pm 0.0180$ | $0.8080 \pm 0.0420$ | $\mathbf{0.8750 \pm 0.0559}$ | $\mathbf{1.10 \pm 0.22}$ |

### Key Empirical Findings:
1. **The Collaborative Gain:** Local-only models suffer catastrophic failure on attacks outside their silo (Client B misses 100% of DoS and Web Attacks; Client A catches only 1.16% of PortScans). Federated learning eliminates these blindspots.
2. **FedProx Regularization:** On extreme attack silos, FedProx ($\mu^*=1.0$) outperforms FedAvg by **+5.7 points** in macro F1 ($0.8750$ vs. $0.8182$) and lifts minority Web Attack recall from 33.47% to **50.77% (+17.3 points)**, winning in 4 of 5 seeds.

---

## 3. Production Security, Privacy & Efficiency Benchmark

| Evaluation Dimension | Configuration | Bandwidth / Privacy | Test F1-Score | Recall | FPR (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Dynamic Quantization** | FP32 (Full Precision) | 85.65 KB | $0.9898 \pm 0.0004$ | $0.9980 \pm 0.0012$ | $1.01 \pm 0.08$ |
| | FP16 (Half Precision) | 46.40 KB (-45.8%) | $0.9898 \pm 0.0004$ | $0.9980 \pm 0.0012$ | $1.01 \pm 0.08$ |
| | **Dynamic INT8** | **32.88 KB (-61.6%)** | $\mathbf{0.9900 \pm 0.0004}$ | $0.9979 \pm 0.0010$ | $\mathbf{0.98 \pm 0.09}$ |
| **Byzantine Poisoning** | 3 Clients Clean (FedAvg) | No Attack | $0.9894 \pm 0.0009$ | $0.9980 \pm 0.0003$ | $1.25 \pm 0.15$ |
| | 3 Clients (33.3% Poisoned, FedAvg)| Label Flip ($y \leftarrow 1-y$) | $0.9738 \pm 0.0085$ | $0.9942 \pm 0.0040$ | $2.36 \pm 0.70$ (Peak: 3.34%) |
| | 3 Clients (33.3% Poisoned, Trimmed)| Trimmed Mean ($\beta=0.2$) | $0.9858 \pm 0.0025$ | $0.9975 \pm 0.0011$ | $1.24 \pm 0.20$ |
| | **3 Clients (33.3% Poisoned, Median)**| **Coordinate Median** | $\mathbf{0.9860 \pm 0.0021}$ | $0.9978 \pm 0.0009$ | $\mathbf{1.21 \pm 0.23}$ |
| **Differential Privacy** | Non-DP Baseline (GroupNorm) | $\epsilon = \infty$ | $0.9851 \pm 0.0032$ | $0.9994 \pm 0.0005$ | $1.08 \pm 0.12$ |
| | DP-SGD ($\sigma=0.5, C=1.0$) | $\epsilon = 26.40$ at $\delta=10^{-5}$ | $0.9548 \pm 0.0013$ | $0.9322 \pm 0.0025$ | $0.85 \pm 0.06$ |
| | **DP-SGD ($\sigma=1.0, C=1.0$)** | $\mathbf{\epsilon = 4.06}$ at $\mathbf{\delta=10^{-5}}$ | $\mathbf{0.9486 \pm 0.0020}$ | $0.9220 \pm 0.0031$ | $\mathbf{0.78 \pm 0.05}$ |
| | DP-SGD ($\sigma=2.0, C=1.0$) | $\mathbf{\epsilon = 1.41}$ at $\delta=10^{-5}$ | $0.9340 \pm 0.0064$ | $0.8961 \pm 0.0082$ | $0.62 \pm 0.04$ |

---

## 4. Team Contribution & 7-Month Task Ownership Matrix

Work is divided equally (33.3% each) with 7 dedicated monthly tasks per member (21 total tasks across the project):

| Phase / Month | Member 1 (Federation & Security Defenses) | Member 2 (Data Engineering, Partitions & Compression) | Member 3 (Client Operations, Privacy & Interactive UI) |
| :--- | :--- | :--- | :--- |
| **Month 1 (W1)** | 5-tuple flow analysis & FedAvg mathematical formulation | Raw data cleaning, infinite/NaN handling, constant feature pruning | Stratified 70/10/20 data split construction & leak-free scaling |
| **Month 2 (W2)** | Logistic Regression & Random Forest baseline models | 4-layer PyTorch MLP implementation (`input_dim -> 128 -> 64 -> 32 -> 1`) | Unified evaluation metrics module (F1, PR-AUC, FPR) & feature importance |
| **Month 3 (W3)** | Server-side weighted FedAvg simulation across communication rounds | Multi-organization partitioner simulating 3 disparate enterprise entities | Flower `NumPyClient` interface (`get_parameters`, `fit`, `evaluate`) |
| **Month 4 (W4)** | Multi-seed synchronization (`[42, 101, 777]`) & cross-split leakage checks | CIC-IDS2017 curation, deduplication, & QuantileTransformer fix (F1: 0.43 -> 0.9883) | Per-attack recall decomposition & isolating silo blindspots |
| **Month 5 (W5)** | FedProx proximal penalty $\frac{\mu}{2}\|w-w_t\|^2$ & validation tuning ($\mu^*=1.0$) | Extreme attack silos partitioner & continuous Dirichlet skew ($\alpha=0.5, 0.1$) | Decentralized scaling privacy benchmark & 5-seed paired differences |
| **Month 6 (W6)** | Byzantine poisoning defense benchmark (Coordinate Median & Trimmed Mean) | Post-training dynamic INT8 quantization (61.6% savings) & Web Attack ablation | Opacus DP-SGD integration with GroupNorm MLP ($\epsilon=4.06$ at $\delta=10^{-5}$) |
| **Month 7 (W7)** | Production Docker containerization suite (`Dockerfile`, `docker-compose.yml`) | Cross-dataset domain generalization evaluation & LaTeX publication tables | Interactive Streamlit demo dashboard (`app/app.py`) & 13 unit tests |

Detailed breakdown available in:
- Markdown: [`reports/weekly_progress_report_months_1_to_7.md`](file:///c:/Users/saifa/.gemini/antigravity/scratch/NetSentry-FL/reports/weekly_progress_report_months_1_to_7.md)
- Word Document: [`reports/NetSentry-FL_Weekly_Progress_Report_Months_1_to_7.docx`](file:///c:/Users/saifa/.gemini/antigravity/scratch/NetSentry-FL/reports/NetSentry-FL_Weekly_Progress_Report_Months_1_to_7.docx)

---

## 5. Quickstart & Reproduction

### 1. Installation
```bash
git clone https://github.com/Saifan-01/NetSentry-FL.git
cd NetSentry-FL
pip install -r requirements.txt
```

### 2. Run Automated Unit Tests (13 Tests Passing in 0.19s)
```bash
pytest tests/
```

### 3. Run Benchmarks
```bash
# Month 4 Baselines:
python src/models/evaluate_real_cicids.py

# Month 5 Heterogeneity & FedProx:
python src/models/evaluate_month5_tightened.py

# Month 6 INT8 Quantization:
python src/federated/quantization_benchmark.py

# Month 6 Byzantine Poisoning Defense:
python src/federated/robust_aggregation.py

# Month 6 Differential Privacy (Opacus DP-SGD):
python src/privacy/evaluate_dp.py

# Month 7 Cross-Dataset Domain Generalization:
python src/models/evaluate_cross_dataset.py
```

### 4. Launch Interactive Streamlit Dashboard
```bash
streamlit run app/app.py
```

### 5. Launch via Docker Compose
```bash
docker-compose up --build
```
Access dashboard at `http://localhost:8501`.

---

## 6. Repository Layout
```
NetSentry-FL/
├── app/
│   └── app.py                             # Interactive Streamlit Demo (6 Tabs, Schema Validator)
├── data/
│   ├── raw/                               # Curated CIC-IDS2017 parquet & CSV sources
│   ├── processed/                         # Leak-free 70/10/20 train/val/test splits
│   └── demo_samples/                      # Synthetic and real test flow payloads for live testing
├── experiments/                           # Output benchmark CSVs across 5 seeds
├── reports/
│   ├── charts/                            # High-resolution PNG evaluation figures
│   ├── research_report.md                 # 15-page comprehensive technical dissertation
│   ├── presentation_slides.md             # 13-slide defense presentation deck
│   ├── latex_tables.tex                   # Publication-ready LaTeX tables for paper submission
│   ├── weekly_progress_report_months_1_to_7.md # Equal 3-member task division
│   └── NetSentry-FL_Weekly_Progress_Report_Months_1_to_7.docx # Formatted Word report
├── src/
│   ├── data/                              # Preprocessing, deduplication, and scaling
│   ├── federated/                         # Client, Server, FedProx, Byzantine & Quantization
│   ├── models/                            # PyTorch NetSentryClassifier & cross-dataset benchmark
│   └── privacy/                           # Opacus DP-SGD trainer & RDP accountant
├── tests/                                 # 13 automated unit tests
├── Dockerfile                             # Production container image
├── docker-compose.yml                     # Multi-service deployment definition
└── requirements.txt                       # Python dependencies
```
