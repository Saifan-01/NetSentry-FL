# NetSentry-FL: Privacy-Preserving Federated Network Intrusion Detection System

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Flower](https://img.shields.io/badge/Flower-Federated%20Learning-ff4b4b.svg)](https://flower.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b.svg)](https://streamlit.io/)

NetSentry-FL is an end-to-end Federated Learning framework for Network Intrusion Detection Systems (NIDS) evaluated on the **CIC-IDS2017** benchmark. It allows multiple simulated organizations (Clients A, B, and C) to collaborate and train a joint deep neural network using **Flower FedAvg** without exchanging raw network flow records.

---

## 🏛️ System Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │          Flower FedAvg Server                │
                    │        Global IDS Model (MLP)                │
                    └───────┬──────────────┬──────────────┬────────┘
                            │              │              │
             Global Weights │              │              │ Global Weights
                   (73.7 KB)│              │              │ (73.7 KB)
                            ▼              ▼              ▼
                     ┌──────────────┐┌──────────────┐┌──────────────┐
                     │   Client A   ││   Client B   ││   Client C   │
                     │  (Bank Silo) ││(Health Silo) ││ (Cloud Silo) │
                     └──────┬───────┘└──────┬───────┘└──────┬───────┘
                            │               │               │
                     Local SGD (2 ep)Local SGD (2 ep)Local SGD (2 ep)
                     [PRIVATE DATA]  [PRIVATE DATA]  [PRIVATE DATA]
                     (Never leaves)  (Never leaves)  (Never leaves)
```

---

## 📊 The Main Result: 3-Seed Comparison Table

All experiments are evaluated on the exact same **8,000 held-out test flows** and averaged over **3 distinct seeds (42, 101, 777)**:

| Setup | Precision | Recall | F1-Score | PR-AUC | False Positive Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Centralized (MLP)** | **1.0000** | **0.9639** | **0.9809** | **1.0000** | **0.0000** |
| **Local-only (Client A - PortScan)** | 1.0000 | 0.3865 | 0.5575 | 1.0000 | 0.0000 |
| **Local-only (Client B - DoS Hulk)** | 1.0000 | 0.7010 | 0.8131 | 1.0000 | 0.0000 |
| **Local-only (Client C - DDoS & FTP)**| 1.0000 | 0.2979 | 0.4435 | 0.9450 | 0.0000 |
| **Federated (IID - FedAvg)** | 1.0000 | 0.6031 | 0.7175 | 1.0000 | 0.0000 |
| **Federated (Non-IID - FedAvg)** | **1.0000** | **0.7373** | **0.8289** | **1.0000** | **0.0000** |

---

## 🚀 Quickstart & Reproducibility

### 1. Installation
```bash
git clone https://github.com/Saifan-01/NetSentry-FL.git
cd NetSentry-FL
pip install -r requirements.txt
```

### 2. Dataset Preparation (Pre-processing & Scaling)
```bash
python src/data/generate_subtle_attacks.py
python src/data/make_dataset.py
```

### 3. Run Centralized Baselines
```bash
python src/models/baseline_classifiers.py
python src/models/train_centralized_pytorch.py
```

### 4. Run the 3-Seed Federated Experiments & Generate Visualizations
```bash
python src/models/evaluate_3seeds.py
python src/utils/generate_charts.py
```

### 5. Launch the Interactive Streamlit Demo
```bash
streamlit run app/app.py
```

---

## 📂 Project Organization
```
NetSentry-FL/
├── app/
│   └── app.py                      # Interactive Streamlit Demo & Evaluation Dashboard
├── configs/                        # Hyperparameter configurations
├── data/
│   ├── raw/                        # Raw / benchmark CIC-IDS2017 flow data
│   └── processed/                  # Normalized, leak-free splits
├── docs/                           # Theoretical notes & Phase 1 documentation
├── experiments/                    # Saved evaluation CSVs and model checkpoints
├── reports/
│   ├── charts/                     # High-res publication evaluation charts
│   ├── research_report.md          # 10-15 page formal technical report
│   └── presentation_slides.md      # 10-slide viva/presentation defense
├── src/
│   ├── data/                       # Ingestion, cleaning, and preprocessing
│   ├── models/                     # PyTorch NetSentryClassifier & 3-seed evaluator
│   ├── federated/                  # Flower NumPyClient and FedAvg simulation runner
│   └── utils/                      # Publication chart generator
├── README.md
└── requirements.txt
```
