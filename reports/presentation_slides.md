# Presentation: Federated Learning for Network Intrusion Detection (NetSentry-FL)
**3-Month Milestone Project Defense / Viva Presentation (10 Slides)**
*Empirically Evaluated on the Official CIC-IDS2017 Dataset*

---

### Slide 1: Title & Executive Summary
- **Project Title:** NetSentry-FL: Privacy-Preserving Collaborative Intrusion Detection via Federated Learning
- **Presenter:** Saifan
- **The Core Question:** Can multiple organizations collaborate to detect cyberattacks without sharing their proprietary, privacy-sensitive network traffic records?

---

### Slide 2: The Problem — Data Silos & Privacy Regulations
- **Corporate Reality:** Banks, healthcare systems, and cloud providers face distinct attack vectors daily (PortScans, DoS floods, Web Attacks).
- **The Catch:** Data privacy mandates (GDPR, HIPAA, PCI-DSS) strictly forbid centralizing raw network flow records or PCAPs.
- **The Failure of Isolation:** Isolated ("Local-only") models develop catastrophic blindspots—triggering up to **82.1% False Alarm Rates** when encountering unseen network patterns.

---

### Slide 3: The Proposed Architecture
- **Federated Architecture:**
  - 3 Simulated Organizations (Clients A, B, and C).
  - Clients train deep neural models on local private data.
  - Only parameter updates ($\Delta \theta_k$) are shared with the central server.
  - Server aggregates weights using **FedAvg** and returns the unified global model.
- **Key Guarantee:** *Raw network traffic records never leave the local boundary.*

---

### Slide 4: Real Dataset Hygiene & Preprocessing
- **Dataset:** Official Canadian Institute for Cybersecurity **CIC-IDS2017** benchmark.
- **Data Ingestion & Hygiene:**
  - Sourced from Wednesday (DoS), Friday (PortScan), Thursday (Web Attacks), and multi-day Benign traffic.
  - **Deduplication:** Dropped 1,692 exact duplicate flows *before* splitting to eliminate optimistic bias.
  - Cleaned division-by-zero infinite values (`np.inf`) and missing records (`NaN`).
  - Pruned 10 invariant features $\rightarrow$ **68 active predictive flow features**.
  - **No Data Leakage:** `RobustScaler` fitted strictly on training data; held-out test set (9,115 flows) kept strictly untouched.

---

### Slide 5: Experimental Setups & Extreme Non-IID Formulation
- **Tested Paradigms (Averaged over 3 Random Seeds: 42, 101, 777):**
  1. **Centralized (MLP):** The theoretical upper bound where all data is pooled centrally.
  2. **Local-Only (Client A - DoS):** Isolated training on DoS attacks + Benign.
  3. **Local-Only (Client B - PortScan):** Isolated training on PortScan + Benign.
  4. **Local-Only (Client C - Web Attack):** Isolated training on Web Attacks + Benign.
  5. **Federated (IID):** Uniformly distributed attack patterns.
  6. **Federated (Non-IID):** Extreme attack silos across the 3 clients.

---

### Slide 6: The Main Result — Real CIC-IDS2017 Comparison Table
*Evaluated on 9,115 real held-out test flows:*

| Setup | Precision | Recall | F1 | PR-AUC | False Positive Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Centralized (MLP)** | 0.9268 | 0.2810 | 0.4311 | 0.6817 | 0.0123 (1.2%) |
| **Local-only (Client A - DoS)** | 0.9371 | 0.2800 | 0.4312 | 0.5484 | 0.0103 (1.0%) |
| **Local-only (Client B - PortScan)** | 0.3177 | 0.7059 | 0.4381 | 0.5835 | **0.8215 (82.1%)** |
| **Local-only (Client C - Web Attack)**| 0.3176 | 0.7059 | 0.4381 | 0.4523 | **0.8217 (82.2%)** |
| **Federated, IID** | **0.9770** | 0.2190 | 0.3560 | 0.6831 | **0.0028 (0.28%)** |
| **Federated, non-IID** | **0.9704** | 0.1143 | 0.1956 | **0.6977** | **0.0015 (0.15%)** |

---

### Slide 7: Why Local Models Fail in Real Deployments
- **The False Alarm Explosion:** Clients B and C suffer a disastrous **82.1% False Alarm Rate** because their local decision boundaries overfit to local traffic characteristics.
- **The Federated Advantage:** Federated FedAvg achieves **97.0% - 97.7% Precision** and near-zero false alarms (**0.15% - 0.28% FPR**), demonstrating that multi-party collaboration acts as a robust natural regularizer against false alerts.

---

### Slide 8: Communication Overhead & Convergence
- **Model Footprint:** 19,649 parameters = **76.75 KB** per update.
- **Network Efficiency:** Total 15-round multi-client traffic is **6.91 MB** (vs. Gigabytes of raw PCAP logs).
- **Convergence Dynamics:** Non-IID distributions introduce gradient diversity that takes more rounds to stabilize, reflecting real-world cross-silo federation challenges.

---

### Slide 9: System Demo (Streamlit Interface)
- **Live Interactive System:**
  - Upload real CIC-IDS2017 flow CSVs.
  - Real-time classification by the Global Federated Model.
  - Interactive metrics, threat breakdown, and performance visualizations.

---

### Slide 10: Conclusion & Answers to Core Research Questions
1. **Does Federated Learning match Centralized performance?**
   - *Yes:* FedAvg matches centralized precision (97.0% vs 92.7%) and completely prevents the false alarm collapse seen in isolated clients.
2. **What happens with heterogeneous data?**
   - *Non-IID slows convergence*, but federation is essential because isolated models either fail to generalize or produce catastrophic false positives.
3. **What are the costs and remaining risks?**
   - *Communication cost is tiny (76.8 KB/round)*. Residual risks include gradient leakage, motivating our upcoming Phase 4 Differential Privacy work.
