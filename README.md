# NetSentry-FL: Federated Learning for Network Intrusion Detection System

## Project Overview
NetSentry-FL is a privacy-preserving Network Intrusion Detection System (NIDS) utilizing Federated Learning (FL) across simulated collaborative organizations without exchanging raw network flow records.

## Project Structure
```
NetSentry-FL/
├── data/
│   ├── raw/                  # Raw CIC-IDS2017 flow CSV files
│   └── processed/            # Cleaned, normalized, and split datasets
├── src/
│   ├── data/                 # Data loading, cleaning, and preprocessing
│   ├── models/               # Centralized baselines and PyTorch architectures
│   ├── federated/            # Flower client, server, and aggregation logic
│   └── utils/                # Metrics, visualization, and logging utilities
├── configs/                  # Experiment configurations (hyperparameters, client splits)
├── experiments/              # Benchmark results, training logs, and metrics
├── notebooks/                # Exploratory Data Analysis (EDA) and visualization
└── docs/                     # Research notes, literature review, and architecture designs
```
