# NetSentry-FL: Privacy-Preserving Federated Network Intrusion Detection System

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Flower](https://img.shields.io/badge/Flower-Federated%20Learning-ff4b4b.svg)](https://flower.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b.svg)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/Dataset-Real%20CIC--IDS2017-green.svg)](https://www.unb.ca/cic/datasets/ids-2017.html)

NetSentry-FL is an end-to-end Federated Learning framework for Network Intrusion Detection Systems (NIDS) evaluated on the real **Canadian Institute for Cybersecurity (CIC-IDS2017)** benchmark dataset. It simulates three collaborative organizations (Clients A, B, and C) training a shared deep neural network using **Flower FedAvg** without exchanging raw network flow records.

---

## System Architecture

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

## The Main Result: Real CIC-IDS2017 3-Seed Comparison Table

All experiments are evaluated on the exact same **7,292 real held-out test flows** (20% held-out test partition) and averaged over **3 distinct seeds (42, 101, 777)** with reported standard deviations:

| Setup | Precision | Recall | F1 | PR-AUC | False Positive Rate (FPR) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Centralized (MLP)** | 0.9503 $\pm$ 0.0120 | 0.2743 $\pm$ 0.0047 | 0.4256 $\pm$ 0.0051 | 0.6842 $\pm$ 0.0112 | 0.0078 $\pm$ 0.0021 (0.78%) |
| **Local-only (Client A - DoS)** | 0.9232 $\pm$ 0.0117 | 0.2775 $\pm$ 0.0018 | 0.4268 $\pm$ 0.0013 | 0.5529 $\pm$ 0.0024 | 0.0125 $\pm$ 0.0021 (1.25%) |
| **Local-only (Client B - PortScan)** | 0.3187 $\pm$ 0.0001 | 0.7096 $\pm$ 0.0004 | 0.4398 $\pm$ 0.0002 | 0.6085 $\pm$ 0.0095 | **0.8222 $\pm$ 0.0000 (82.22%)** |
| **Local-only (Client C - Web Attack)**| 0.3189 $\pm$ 0.0002 | 0.7096 $\pm$ 0.0004 | 0.4400 $\pm$ 0.0002 | 0.4416 $\pm$ 0.0070 | **0.8215 $\pm$ 0.0010 (82.15%)** |
| **Federated, IID** | **0.9635 $\pm$ 0.0090** | 0.2731 $\pm$ 0.0048 | 0.4255 $\pm$ 0.0050 | 0.6848 $\pm$ 0.0037 | **0.0056 $\pm$ 0.0015 (0.56%)** |
| **Federated, non-IID** | 0.6111 $\pm$ 0.4374 | 0.0009 $\pm$ 0.0008 | 0.0018 $\pm$ 0.0016 | 0.6609 $\pm$ 0.0848 | **0.0001 $\pm$ 0.0001 (0.01%)** |

---

## Centralized Baseline Models (Pooled Data)

| Model | Accuracy | Precision | Recall | F1-Score | PR-AUC | False Alarm Rate (FPR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 63.00% | 47.36% | 47.17% | 0.4726 | 0.6533 | 28.42% |
| **Random Forest (100 Trees)** | 99.81% | 99.84% | 99.61% | 0.9973 | 1.0000 | 0.08% |
| **Centralized PyTorch MLP** | 74.18% | 95.95% | 27.70% | 0.4299 | 0.6920 | 0.63% |

---

## Quickstart & Reproducibility

### 1. Installation
```bash
git clone https://github.com/Saifan-01/NetSentry-FL.git
cd NetSentry-FL
pip install -r requirements.txt
```

### 2. Dataset Curation (Deduplication & 70/10/20 Stratified Split)
```bash
python src/data/curate_real_cicids2017.py
```

### 3. Run Centralized Baselines
```bash
python src/models/baseline_classifiers.py
```

### 4. Run 3-Seed Federated Experiments & Generate Visualizations
```bash
python src/models/evaluate_real_cicids.py
python src/utils/generate_charts.py
```

### 5. Run Automated Unit Tests
```bash
python -m unittest discover tests
```

### 6. Launch the Interactive Streamlit Demo
```bash
streamlit run app/app.py
```

---

## Project Structure
```
NetSentry-FL/
├── app/
│   └── app.py                      # Interactive Streamlit Demo & Evaluation Dashboard
├── configs/                        # Hyperparameter configurations
├── data/
│   ├── raw/                        # Real CIC-IDS2017 flow data parquets & curated CSV
│   ├── processed/                  # Normalized, leak-free 70/10/20 splits
│   └── demo_samples/               # Pre-packaged ready-to-test flow CSVs
├── docs/                           # Theoretical notes & Phase 1 documentation
├── experiments/                    # Saved evaluation CSVs and model checkpoints
├── reports/
│   ├── charts/                     # Publication evaluation charts (PNG)
│   ├── execution_and_fix_log.md    # Detailed execution and research changelog
│   ├── research_report.md          # 10-15 page formal technical report
│   └── presentation_slides.md      # 10-slide viva/presentation defense
├── src/
│   ├── data/                       # Ingestion, cleaning, deduplication, and preprocessing
│   ├── models/                     # PyTorch NetSentryClassifier, baselines & 3-seed evaluator
│   ├── federated/                  # Flower NumPyClient and FedAvg simulation runner
│   └── utils/                      # Publication chart generator
├── tests/                          # Automated unit tests for preprocessing & partitioner
├── README.md
└── requirements.txt
```
