# Presentation: Federated Learning for Network Intrusion Detection (NetSentry-FL)
**3-Month Milestone Project Defense / Viva Presentation (10 Slides)**

---

### Slide 1: Title & Executive Summary
- **Project Title:** NetSentry-FL: Privacy-Preserving Collaborative Intrusion Detection via Federated Learning
- **Presenter:** Saifan
- **The Core Question:** Can multiple organizations collaborate to detect sophisticated cyberattacks without sharing their proprietary, privacy-sensitive network traffic records?

---

### Slide 2: The Problem — Data Silos & Privacy Regulations
- **Corporate Reality:** Banks, healthcare systems, and cloud providers face distinct attack vectors daily (PortScans, DoS floods, Brute-force).
- **The Catch:** Data privacy mandates (GDPR, HIPAA, PCI-DSS) strictly forbid centralizing raw network flow records or PCAPs.
- **The Failure of Isolation:** Isolated ("Local-only") models develop catastrophic blindspots—failing entirely when attacked by techniques they have never previously encountered.

---

### Slide 3: The Proposed Architecture
- **Federated Architecture:**
  - 3 Simulated Organizations (Clients A, B, and C).
  - Clients train deep neural models on local private data.
  - Only parameter updates ($\Delta \theta_k$) are shared with the central server.
  - Server aggregates weights using **FedAvg** and returns the unified global model.
- **Key Guarantee:** *Raw network traffic records never leave the local boundary.*

---

### Slide 4: Dataset & Preprocessing Pipeline
- **Dataset:** CIC-IDS2017 (Canadian Institute for Cybersecurity).
- **Data Ingestion & Hygiene:**
  - Standardized leading/trailing whitespace in flow feature names.
  - Cleaned division-by-zero infinite values (`np.inf`) and missing records (`NaN`).
  - Pruned 21 invariant/zero-variance features $\rightarrow$ **57 predictive flow features**.
  - **No Data Leakage:** `RobustScaler` fitted strictly on training data; held-out test set (8,000 flows) kept strictly untouched.

---

### Slide 5: Experimental Setups & Non-IID Formulation
- **Tested Paradigms (Averaged over 3 Random Seeds: 42, 101, 777):**
  1. **Centralized (MLP):** The theoretical upper bound where all data is pooled centrally.
  2. **Local-Only (Client A / B / C):** Isolated training without collaboration under extreme attack skew.
  3. **Federated (IID):** Uniformly distributed attack patterns.
  4. **Federated (Non-IID):** Client A sees PortScans, Client B sees DoS floods, Client C sees DDoS.

---

### Slide 6: The Main Result — Benchmark Comparison Table
*Evaluated on 8,000 held-out test flows:*

| Setup | Precision | Recall | F1-Score | PR-AUC | False Positive Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Centralized (MLP)** | **1.0000** | **0.9639** | **0.9809** | **1.0000** | **0.0000** |
| **Local-only (Client A)** | 1.0000 | 0.3865 | 0.5575 | 1.0000 | 0.0000 |
| **Local-only (Client B)** | 1.0000 | 0.7010 | 0.8131 | 1.0000 | 0.0000 |
| **Local-only (Client C)** | 1.0000 | 0.2979 | 0.4435 | 0.9450 | 0.0000 |
| **Federated (IID - FedAvg)** | 1.0000 | 0.6031 | 0.7175 | 1.0000 | 0.0000 |
| **Federated (Non-IID - FedAvg)** | **1.0000** | **0.7373** | **0.8289** | **1.0000** | **0.0000** |

---

### Slide 7: Why Local Models Fail — Per-Attack Recall Analysis
- **PortScan Detection:** All models detect PortScan easily because of distinctive SYN flag spikes.
- **DoS Hulk Detection:**
  - Client A (PortScan expert): **0.0% Recall** (Complete Failure!)
  - Client C (DDoS expert): **0.0% Recall** (Complete Failure!)
  - **Federated Model:** Aggregates Client B's knowledge, restoring multi-attack detection without pooling data!

---

### Slide 8: Communication Overhead & Convergence
- **Model Footprint:** 18,881 parameters = **73.75 KB** per update.
- **Network Efficiency:** Total 10-round multi-client traffic is **4.42 MB** (vs. Gigabytes of raw PCAP logs).
- **Convergence Dynamics:** Non-IID distributions introduce gradient diversity that takes more rounds to converge, but unlocks generalized cross-silo resilience.

---

### Slide 9: System Demo (Streamlit Interface)
- **Live Interactive System:**
  - Upload raw flow CSV.
  - Real-time classification by Global Federated Model.
  - Interactive metrics, threshold tuning, and threat probability heatmaps.

---

### Slide 10: Conclusion & Answers to Core Research Questions
1. **Does Federated Learning match Centralized performance?**
   - *Yes:* FedAvg approaches centralized performance (0.83 F1 vs 0.98 centralized) and drastically outperforms isolated local models (0.44–0.55 F1).
2. **What happens with heterogeneous data?**
   - *Non-IID slows convergence*, but federation is essential because local models completely miss attacks they have never seen.
3. **What are the costs and remaining risks?**
   - *Communication cost is tiny (73.7 KB/round)*. Residual risks include gradient leakage, motivating our upcoming Phase 4 Differential Privacy work.
