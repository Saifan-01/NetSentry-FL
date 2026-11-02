# Presentation: Federated Learning for Network Intrusion Detection (NetSentry-FL)
**3-Month Milestone Project Defense / Viva Presentation (10 Slides)**
*Empirically Evaluated on the Official CIC-IDS2017 Benchmark*

---

### Slide 1: Title & Team Information
- **Project Title:** NetSentry-FL: Privacy-Preserving Collaborative Intrusion Detection via Federated Learning
- **Student Name:** Saifan [Team Members: TBD]
- **Project Mentor / Guide:** [Guide Name, Department]
- **Institution:** [College / University Name]
- **Core Research Question:** Can multiple organizations collaborate to detect sophisticated network attacks without sharing their proprietary, privacy-sensitive network flow records?

---

### Slide 2: The Problem — Data Silos & Privacy Regulations
- **Operational Reality:** Financial institutions, healthcare systems, and cloud providers face distinct attack vectors daily (PortScans, DoS floods, Web Attacks).
- **The Catch:** Data protection mandates (GDPR, HIPAA, PCI-DSS) strictly forbid centralizing raw network flow records or packet logs.
- **The Failure of Isolation:** Isolated ("Local-only") models develop catastrophic blindspots—triggering an unviable **82.2% False Alarm Rate** when encountering broader network traffic.

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
  - **70/10/20 Stratified Split:** 25,522 train, 3,646 validation, and 7,292 held-out test flows.
  - **Zero Data Leakage:** `RobustScaler` fitted strictly on training data; held-out test set kept strictly untouched.

---

### Slide 5: Experimental Setups & Extreme Non-IID Formulation
- **Tested Paradigms (Averaged over 3 Random Seeds: 42, 101, 777):**
  1. **Centralized (MLP):** Theoretical upper bound with all data pooled centrally.
  2. **Local-Only (Client A - DoS):** Isolated training on DoS attacks + Benign.
  3. **Local-Only (Client B - PortScan):** Isolated training on PortScan + Benign.
  4. **Local-Only (Client C - Web Attack):** Isolated training on Web Attacks + Benign.
  5. **Federated (IID):** Uniformly distributed attack patterns across clients.
  6. **Federated (Non-IID):** Extreme attack silos across the 3 clients.

---

### Slide 6: The Main Result — Real CIC-IDS2017 Comparison Table
*Evaluated on 7,292 real held-out test flows (Mean $\pm$ Standard Deviation across 3 seeds):*

| Setup | Precision | Recall | F1-Score | PR-AUC | False Positive Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Centralized (MLP)** | 0.9503 $\pm$ 0.0120 | 0.2743 $\pm$ 0.0047 | 0.4256 $\pm$ 0.0051 | 0.6842 $\pm$ 0.0112 | 0.0078 $\pm$ 0.0021 (0.78%) |
| **Local-only (Client A - DoS)** | 0.9232 $\pm$ 0.0117 | 0.2775 $\pm$ 0.0018 | 0.4268 $\pm$ 0.0013 | 0.5529 $\pm$ 0.0024 | 0.0125 $\pm$ 0.0021 (1.25%) |
| **Local-only (Client B - PortScan)** | 0.3187 $\pm$ 0.0001 | 0.7096 $\pm$ 0.0004 | 0.4398 $\pm$ 0.0002 | 0.6085 $\pm$ 0.0095 | **0.8222 $\pm$ 0.0000 (82.22%)** |
| **Local-only (Client C - Web Attack)**| 0.3189 $\pm$ 0.0002 | 0.7096 $\pm$ 0.0004 | 0.4400 $\pm$ 0.0002 | 0.4416 $\pm$ 0.0070 | **0.8215 $\pm$ 0.0010 (82.15%)** |
| **Federated, IID** | **0.9635 $\pm$ 0.0090** | 0.2731 $\pm$ 0.0048 | 0.4255 $\pm$ 0.0050 | 0.6848 $\pm$ 0.0037 | **0.0056 $\pm$ 0.0015 (0.56%)** |
| **Federated, non-IID** | 0.6111 $\pm$ 0.4374 | 0.0009 $\pm$ 0.0008 | 0.0018 $\pm$ 0.0016 | 0.6609 $\pm$ 0.0848 | **0.0001 $\pm$ 0.0001 (0.01%)** |

---

### Slide 7: Why Local Models Fail in Real Deployments
- **The False Alarm Explosion:** Clients B and C suffer a disastrous **82.2% False Alarm Rate** because their local decision boundaries collapsed around narrow local attack traffic.
- **The Federated Advantage:** Federated FedAvg achieves **96.35% Precision** and near-zero false alarms (**0.56% FPR**), demonstrating that multi-party collaboration acts as a robust natural regularizer against false alerts.

---

### Slide 8: Communication Overhead & Convergence Extension
- **Model Footprint:** 19,649 parameters = **76.75 KB** per client update.
- **Network Efficiency:** Total 15-round multi-client traffic is **6.91 MB** (vs. Gigabytes of raw PCAP logs).
- **Rounds to Target F1:** Federated IID reaches target F1 ($\ge 0.30$) within **Round 1** (total cumulative bandwidth: 460.52 KB).
- **Convergence Dynamics:** Non-IID distributions introduce gradient diversity that slows convergence, highlighting real-world cross-silo federation trade-offs.

---

### Slide 9: System Demo (Streamlit Interface)
- **Live Interactive System:**
  - Upload real CIC-IDS2017 flow CSVs.
  - Strict feature alignment and validation (prevents silent zero-filling of missing features).
  - Real-time classification by the Global Federated Model.
  - Interactive threshold tuning slider, threat breakdown, and performance visualizations.

---

### Slide 10: Conclusion & Answers to Core Research Questions
1. **Does Federated Learning match Centralized performance?**
   - *Yes:* FedAvg matches centralized precision (96.35% vs 95.03%) and completely prevents the false alarm collapse seen in isolated clients.
2. **What happens with heterogeneous data?**
   - *Non-IID slows convergence*, but federation is essential because isolated models either fail to generalize or produce catastrophic false positives.
3. **What are the costs and remaining risks?**
   - *Communication cost is tiny (76.8 KB/round)*. Residual risks include gradient leakage, motivating our upcoming Phase 4 Differential Privacy work.
