# NetSentry-FL: Comprehensive Progress Report (Months 1 to 7)

**Project:** Federated Learning for Network Intrusion Detection  
**Repository:** `https://github.com/Saifan-01/NetSentry-FL.git`  
**Team Members:** Member 1, Member 2, Member 3 *(Replace with real student names)*  

---

## 1. Master Contribution Matrix (Months 1 to 7)

Work is strictly divided such that every member owns one major engineering/research task per month, ensuring an equal 33.3% contribution across all phases of the project:

| Month / Phase | Member 1 (Federation & Security Defenses) | Member 2 (Data Engineering, Partitions & Compression) | Member 3 (Client Training, Privacy & Dashboard) |
| :--- | :--- | :--- | :--- |
| **Month 1 (Week 1)**<br>Research & Foundations | Network flow representations (5-tuple), flow-level encrypted traffic analysis, FedAvg mathematical formulation. | Raw data cleaning, header whitespace trimming, infinite/NaN handling, constant column pruning (16 features dropped). | Stratified 70/10/20 data split construction, initial scaler fitting on train split only. |
| **Month 2 (Week 2)**<br>Centralized Baselines | Classical ML baselines: Logistic Regression (balanced weights) and Random Forest (100 trees). | Centralized PyTorch MLP architecture implementation (4 layers, BatchNorm, LeakyReLU, Dropout). | Metric evaluation module (Precision, Recall, F1, PR-AUC, FPR) and feature importance ranking. |
| **Month 3 (Week 3)**<br>Federated Prototype | Server-side weighted FedAvg simulation, round-by-round aggregation history over synthetic data. | Organizational multi-client partitioner for 3 entities (Financial, Healthcare, Cloud) with initial splits. | Flower `NumPyClient` implementation (`fit`, `evaluate`) and isolated local-only training baselines. |
| **Month 4 (Week 4)**<br>Real CIC-IDS2017 & Trustworthy Baseline | Re-evaluating baselines on real data, multi-seed synchronization (`[42, 101, 777]`), zero-leakage cross-split audits. | Real dataset curation (25,522 train, 7,292 test flows), deduplication before split, QuantileTransformer calibration (MLP F1: 0.43 -> 0.9883). | Per-attack recall decomposition on real traffic, identifying and documenting local silo blindspots. |
| **Month 5 (Week 5)**<br>Heterogeneity & FedProx | FedProx algorithm implementation, proximal penalty term, validation hyperparameter tuning ($\mu^*=1.0$). | Extreme cross-silo partitioner by attack type (DoS, PortScan, Web Attacks) and continuous Dirichlet skew ($\alpha=0.5, 0.1$). | Decentralized feature scaling privacy benchmark (local quantile vs parameterless $\log(1+x)$) and 5-seed paired differences. |
| **Month 6 (Week 6)**<br>Privacy, Efficiency & Robustness | Byzantine and label-flipping poisoning defense benchmark (FedAvg vs Coordinate Median vs Trimmed Mean). | Post-training model quantization benchmark (FP32 vs FP16 vs INT8, 61.6% bandwidth reduction) and Web Attack ablation. | Formal Differential Privacy via Opacus DP-SGD ($\sigma \in \{0.5, 1.0, 2.0\}$, clipping $C=1.0$, GroupNorm MLP, $\epsilon=4.06$). |
| **Month 7 (Week 7)**<br>Packaging, Demo & Submission | Docker containerization (`Dockerfile`, `docker-compose.yml`), Flower server orchestration, and Git release tagging (`v1.0`). | Cross-dataset domain generalization evaluation, final metrics consolidation, and publication table formatting. | Interactive Streamlit demo dashboard (`app/app.py` with 6 tabs, live schema validator, inference slider) and 13 unit tests. |

---

## 2. Detailed Monthly Progress & Ownership Breakdown

### Month 1: Research, Foundations & Initial Preprocessing
**Goal:** Establish problem formulation, mathematical foundation, and leak-free flow dataset.

| Member | Individual Contribution | Core Output / Script |
| :--- | :--- | :--- |
| **Member 1** | Documented flow-level network representation (5-tuple), advantages for encrypted high-throughput enterprise networks, FedAvg mathematical formulation, and model-update leakage threat vectors. | `reports/phase1_research_and_foundations.md` |
| **Member 2** | Fixed CIC-IDS2017 CSV header trailing whitespace, scrubbed infinite and NaN values in flow rate metrics, mapped multi-class attacks to binary labels (`0 = Benign`, `1 = Attack`), and pruned 16 constant/invariant columns leaving 62 valid features. | `src/data/preprocess.py` |
| **Member 3** | Designed the stratified 70/10/20 train/validation/test split and fit the scaling transform strictly on training flows to prevent test-set distribution leakage. | `src/data/make_dataset.py` |

- **Milestone:** Verified clean dataset pipeline with zero data leakage.

---

### Month 2: Centralized Machine Learning Baselines
**Goal:** Establish rigorous centralized upper bounds before federating models.

| Member | Individual Contribution | Core Output / Script |
| :--- | :--- | :--- |
| **Member 1** | Trained and evaluated Logistic Regression with inverse class frequency weights and Random Forest (100 estimators) on pooled data. | `src/models/baseline_classifiers.py` |
| **Member 2** | Implemented the centralized 4-layer PyTorch MLP (`input_dim -> 128 -> 64 -> 32 -> 1`) with LeakyReLU activations and dropout regularization. | `src/models/train_centralized_pytorch.py` |
| **Member 3** | Engineered the unified evaluation module computing Precision, Recall, F1, PR-AUC, and False Positive Rate (FPR), and ran feature importance analysis. | `src/models/evaluate_metrics.py` |

- **Milestone:** Established initial centralized performance metrics.

---

### Month 3: Federated Learning Prototype Architecture
**Goal:** Construct an end-to-end 3-client federated learning system using Flower.

| Member | Individual Contribution | Core Output / Script |
| :--- | :--- | :--- |
| **Member 1** | Implemented server-side weighted FedAvg aggregation and simulated 5 communication rounds, evaluating global test accuracy after every round. | `src/federated/server.py` |
| **Member 2** | Built the synthetic organizational partitioner simulating 3 disparate enterprise entities (Financial Institution, Healthcare System, Cloud Provider). | `src/federated/dataset_partitioner.py` |
| **Member 3** | Implemented the Flower `NumPyClient` interface (`get_parameters`, `fit`, `evaluate`) and executed isolated local-only baselines. | `src/federated/client.py` |

- **Milestone:** Working prototype demonstrated in simulation; revealed synthetic data limitations (100% scores across models), prompting transition to real benchmark data.

---

### Month 4: Real CIC-IDS2017 Benchmark & Trustworthy Baseline
**Goal:** Replace synthetic data with authentic CIC-IDS2017 network traffic and fix the centralized MLP baseline.

| Member | Individual Contribution | Core Output / Script |
| :--- | :--- | :--- |
| **Member 1** | Reran centralized and federated baselines over 3 random seeds (`[42, 101, 777]`), ensuring seeds were fixed, and conducted strict duplicate leakage checks across split boundaries. | `src/models/evaluate_real_cicids.py` |
| **Member 2** | Curated real CIC-IDS2017 traffic (25,522 train, 3,646 val, 7,292 test flows), deduplicated prior to splitting, and replaced heavy-tailed RobustScaler with Gaussian `QuantileTransformer`, resolving the MLP baseline collapse (F1 surged from 0.4299 to **0.9883**). | `src/data/curate_real_cicids2017.py`, `data/processed/` |
| **Member 3** | Conducted per-attack recall breakdown on real test flows, proving that local models have critical blindspots (Client B achieved 0.0% recall on DoS and Web Attacks; Client A caught 1.16% of PortScans). | `experiments/per_attack_recall.csv`, `reports/charts/per_attack_recall.png` |

- **Milestone:** Achieved an honest, reproducible real-data baseline table where Centralized MLP (F1 0.9883) matches Random Forest (F1 0.9973). Tagged `v0.4-month4`.

---

### Month 5: Heterogeneity, Client Drift & FedProx Regularization
**Goal:** Solve non-IID cross-silo divergence and protect minority attack classes.

| Member | Individual Contribution | Core Output / Script |
| :--- | :--- | :--- |
| **Member 1** | Implemented the FedProx proximal penalty $\frac{\mu}{2} \sum \|w - w_t\|^2$ in client and server training, and executed a 12-configuration validation search identifying $\mu^*=1.0$ as optimal on the held-out validation set. | `src/federated/client.py`, `experiments/validation_tuning_grid.csv` |
| **Member 2** | Built the non-IID partitioner generating: (1) Extreme Attack-Type Cross-Silos (Client A = DoS, Client B = PortScan, Client C = Web Attacks); and (2) Continuous Dirichlet label skew ($\alpha=0.5$ and $\alpha=0.1$). | `src/federated/partitioners.py`, `reports/charts/dirichlet_and_silo_distributions.png` |
| **Member 3** | Evaluated decentralized scaling privacy (Local QuantileTransformer achieved F1 0.8425 vs. Pooled 0.8521, proving raw feature pooling is unnecessary) and analyzed 5-seed paired differences (FedProx won in 4 of 5 seeds). | `src/federated/scaling_privacy_analysis.py`, `experiments/scaling_privacy_comparison.csv` |

- **Milestone:** Demonstrated collaboration benefits over silos; showed FedProx recovers Web Attack recall from 33.47% to 50.77% (+17.3 points) and lifts extreme silo F1 from 0.8182 to 0.8750 (+5.7 points). Tagged `v0.5-month5`.

---

### Month 6: Advanced Security, Compression & Formal Privacy
**Goal:** Implement production security extensions: model compression, Byzantine poisoning defense, and Differential Privacy.

| Member | Individual Contribution | Core Output / Script |
| :--- | :--- | :--- |
| **Member 1** | Implemented Byzantine robustness against 1 malicious label-flipping client ($y \leftarrow 1 - y$); demonstrated that Coordinate-wise Median and Trimmed Mean suppress false alarm rate from 2.36% back to **1.21%** (3 clients) and from 2.11% back to **1.26%** (5 clients). | `src/federated/robust_aggregation.py`, `reports/charts/poisoning_robustness.png` |
| **Member 2** | Engineered post-training quantization (FP32 vs FP16 vs dynamic INT8), reducing model transmission footprint from 85.65 KB to **32.88 KB (61.6% bandwidth savings)** with zero loss in F1 (0.9900 vs 0.9898), and analyzed Web Attack learning vs. forgetting dynamics. | `src/federated/quantization_benchmark.py`, `src/federated/web_attack_ablation.py` |
| **Member 3** | Integrated Opacus DP-SGD with noise multiplier sweep ($\sigma \in \{0.5, 1.0, 2.0\}$), clipping norm $C=1.0$, and fixed $\delta=10^{-5}$; redesigned the network with `GroupNorm` to ensure sample independence, achieving $\mathbf{\epsilon=4.06}$ at F1 0.9486. | `src/privacy/evaluate_dp.py`, `reports/charts/dp_epsilon_vs_f1.png` |

- **Milestone:** Achieved rigorous formal DP bounds, verified Byzantine defense, and achieved 61.6% bandwidth compression. Tagged `v0.6-month6`.

---

### Month 7: Packaging, Demonstration & Final Defense
**Goal:** Deliver a production-packaged, reproducible system with interactive UI and comprehensive defense materials.

| Member | Individual Contribution | Core Output / Script |
| :--- | :--- | :--- |
| **Member 1** | Authored the production containerization suite (`Dockerfile`, `docker-compose.yml`), orchestrating the Flower aggregation server and clients, and managed Git release tagging (`v1.0-final`). | `Dockerfile`, `docker-compose.yml`, GitHub release |
| **Member 2** | Built the cross-dataset domain generalization evaluation (zero-shot transfer testing on out-of-distribution traffic) and compiled the comprehensive LaTeX comparison tables for the final dissertation. | `src/models/evaluate_cross_dataset.py`, `experiments/cross_dataset_results.csv` |
| **Member 3** | Developed the interactive Streamlit demonstration dashboard (`app/app.py`) featuring 6 tabs, live CSV flow uploading, schema validation, threat decision slider, and implemented the 13 automated unit tests. | `app/app.py`, `tests/test_aggregators.py`, `tests/test_partitioner.py` |

- **Milestone:** Full reproducible repository ready for project defense and external evaluation.

---

## 3. Team Member Viva / Defense Preparation Guide

Each member should be prepared to defend their dedicated engineering domain during viva questioning:

- **Member 1 Questions:**
  - *Why does FedProx stabilize non-IID cross-silo training compared to standard FedAvg?*  
    *Answer:* The proximal regularization term $\frac{\mu}{2} \|w - w_t\|^2$ penalizes local client parameters from drifting in conflicting orthogonal directions, anchoring updates to the global trajectory.
  - *How does Coordinate Median neutralize a 33.3% or 20.0% label-flipping poisoning attack?*  
    *Answer:* Coordinate-wise median evaluates parameter distributions independently per dimension, discarding poisoned outlier gradients submitted by malicious nodes.
- **Member 2 Questions:**
  - *Why was dynamic INT8 quantization able to reduce model payload size by 61.6% without degrading F1 score?*  
    *Answer:* Over-parameterized neural weights exhibit redundant precision; 8-bit dynamic quantization preserves relative weight magnitude and decision boundaries while dropping memory from 4 bytes to 1 byte per weight.
  - *What causes the Web Attack minority class collapse under sample-weighted FedAvg?*  
    *Answer:* Web Attacks constitute only 16.7% of the total attack flows in training. Under sample-weighted FedAvg, Client C's update mass is diluted by majority clients (DoS and PortScan), causing Web Attack weights to be overwritten in parameter space.
- **Member 3 Questions:**
  - *Why did you replace `BatchNorm1d` with `GroupNorm` when implementing DP-SGD?*  
    *Answer:* BatchNorm normalizes across the mini-batch, coupling samples and violating per-sample gradient independence required for individual gradient norm clipping in differential privacy. GroupNorm normalizes channels per instance independently of batch size.
  - *What does the $(\epsilon = 4.06, \delta = 10^{-5})$ differential privacy guarantee protect in practice?*  
    *Answer:* It guarantees that an untrusted server or eavesdropper intercepting model updates cannot statistically infer whether a specific confidential flow record was part of a participating organization's local training database.
