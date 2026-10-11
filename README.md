# NetSentry-FL: Privacy-Preserving Federated Network Intrusion Detection System

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Flower](https://img.shields.io/badge/Flower-Federated%20Learning-ff4b4b.svg)](https://flower.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b.svg)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/Dataset-Real%20CIC--IDS2017-green.svg)](https://www.unb.ca/cic/datasets/ids-2017.html)

NetSentry-FL is an end-to-end Federated Learning framework for Network Intrusion Detection Systems (NIDS) evaluated on the real **Canadian Institute for Cybersecurity (CIC-IDS2017)** benchmark dataset. It allows multiple simulated organizations (Clients A, B, and C) to collaborate and train a joint deep neural network using **Flower FedAvg** without exchanging raw network flow records.

---

## 🏛️ System Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │          Flower FedAvg Server                │
                    │        Global IDS Model (MLP)                │
                    └───────┬──────────────┬──────────────┬────────┘
                            │              │              │
             Global Weights │              │              │ Global Weights
                   (76.8 KB)│              │              │ (76.8 KB)
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

## 📊 The Main Result: Real CIC-IDS2017 3-Seed Comparison Table

All experiments are evaluated on the exact same **9,115 real held-out test flows** and averaged over **3 distinct seeds (42, 101, 777)**:

| Setup | Precision | Recall | F1 | PR-AUC | False Positive Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Centralized (MLP)** | 0.9268 | 0.2810 | 0.4311 | 0.6817 | 0.0123 (1.2%) |
| **Local-only (Client A - DoS)** | 0.9371 | 0.2800 | 0.4312 | 0.5484 | 0.0103 (1.0%) |
| **Local-only (Client B - PortScan)** | 0.3177 | 0.7059 | 0.4381 | 0.5835 | **0.8215 (82.1%)** |
| **Local-only (Client C - Web Attack)**| 0.3176 | 0.7059 | 0.4381 | 0.4523 | **0.8217 (82.2%)** |
| **Federated, IID** | **0.9770** | 0.2190 | 0.3560 | 0.6831 | **0.0028 (0.28%)** |
| **Federated, non-IID** | **0.9704** | 0.1143 | 0.1956 | **0.6977** | **0.0015 (0.15%)** |

---

## 🚀 Quickstart & Reproducibility

### 1. Installation
```bash
git clone https://github.com/Saifan-01/NetSentry-FL.git
cd NetSentry-FL
pip install -r requirements.txt
```

### 2. Dataset Preparation (Deduplication & Preprocessing)
```bash
python src/data/curate_real_cicids2017.py
```

### 3. Run Centralized Baselines
```bash
python src/models/baseline_classifiers.py
python src/models/train_centralized_pytorch.py
```

### 4. Run the 3-Seed Federated Experiments & Generate Visualizations
```bash
python src/models/evaluate_real_cicids.py
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
│   ├── raw/                        # Real CIC-IDS2017 flow data parquets & curated CSV
│   └── processed/                  # Normalized, leak-free splits
├── docs/                           # Theoretical notes & Phase 1 documentation
├── experiments/                    # Saved evaluation CSVs and model checkpoints
├── reports/
│   ├── charts/                     # High-res publication evaluation charts
│   ├── research_report.md          # 10-15 page formal technical report
│   └── presentation_slides.md      # 10-slide viva/presentation defense
├── src/
│   ├── data/                       # Ingestion, cleaning, deduplication, and preprocessing
│   ├── models/                     # PyTorch NetSentryClassifier & 3-seed evaluator
│   ├── federated/                  # Flower NumPyClient and FedAvg simulation runner
│   └── utils/                      # Publication chart generator
├── README.md
└── requirements.txt
```
