# Phase 1: Research, Foundations & Threat Model

## 1. Network Traffic & Intrusion Detection Foundations (Week 1)

### 1.1 Network Traffic Representations: Packets vs. Network Flows
In modern high-speed corporate networks (10Gbps+), inspecting every individual packet payload via Deep Packet Inspection (DPI) causes excessive computational latency and violates user privacy (due to TLS/HTTPS encryption). Therefore, modern Network Intrusion Detection Systems (NIDS) analyze **bidirectional network flows**.

A **flow** is defined as a sequence of packets sharing a unique **5-Tuple**:
1. **Source IP Address** (`src_ip`)
2. **Destination IP Address** (`dst_ip`)
3. **Source Port** (`src_port`)
4. **Destination Port** (`dst_port`)
5. **Transport Protocol** (`protocol`: TCP, UDP, ICMP)

Flow features aggregate traffic behavior across time, such as:
- Flow duration
- Forward and backward packet lengths and inter-arrival times (IAT)
- TCP flags (SYN, ACK, FIN, RST, PSH, URG)
- Packet flow rates (packets/sec, bytes/sec)
- Sub-flow metrics

### 1.2 Intrusion Detection Paradigms
- **Signature-based Detection (e.g., Snort, Suricata):** Matches known attack patterns/hashes. Fast and precise for known vulnerabilities, but completely blind to zero-day attacks and polymorphic traffic.
- **Anomaly/ML-based Detection (NIDS):** Learns statistical distributions and structural properties of benign vs. malicious traffic. Can generalize to novel variations, but requires careful handling of false positive rates (FPR) and class imbalance.

---

## 2. Federated Learning Foundations (Week 2)

### 2.1 The Multi-Organization Problem
Consider three independent organizations (Client A: Financial Institution, Client B: Healthcare System, Client C: Cloud Provider).
- Each faces distinct network intrusion attempts (e.g., DDoS, port scans, brute force).
- Strict privacy laws (GDPR, HIPAA, PCI-DSS) and confidentiality policies forbid pooling raw traffic logs into a central cloud repository.
- Training models in isolation ("Local-Only") produces weak detectors with poor out-of-distribution generalization.

### 2.2 Federated Learning Architecture & FedAvg
Federated Learning enables collaborative training without sharing raw data:
1. **Initialization:** The global aggregation server initializes model weights $W_0$.
2. **Distribution:** The server broadcasts $W_t$ to selected participating clients $K$.
3. **Local Training:** Each client $k \in K$ trains on its private local dataset $\mathcal{D}_k$ for $E$ epochs using local SGD:
   $$w_{t+1}^k = w_t^k - \eta \nabla \mathcal{L}(w_t^k; \mathcal{D}_k)$$
4. **Model Upload:** Clients transmit only their updated weights (or weight differences $\Delta w_t^k = w_{t+1}^k - w_t^k$) back to the server.
5. **FedAvg Server Aggregation:** The server computes the weighted average based on sample sizes $n_k = |\mathcal{D}_k|$ and $n = \sum_{k} n_k$:
   $$W_{t+1} = \sum_{k=1}^K \frac{n_k}{n} w_{t+1}^k$$
6. **Iteration:** The cycle repeats across communication rounds $T$.

```
       ┌───────────────────────────────┐
       │   Federated Learning Server   │
       │     (Global Model FedAvg)     │
       └──────┬──────────┬──────────┬──┘
              │          │          │
    Weights W │          │          │
              ▼          ▼          ▼
       ┌──────────┐ ┌──────────┐ ┌──────────┐
       │ Client A │ │ Client B │ │ Client C │
       │ (Bank)   │ │ (Health) │ │ (Cloud)  │
       └────┬─────┘ └────┬─────┘ └────┬─────┘
            │            │            │
       [Data A]     [Data B]     [Data C]
     (Kept Local) (Kept Local) (Kept Local)
```

### 2.3 Security and Privacy Considerations
- **Raw Data Privacy:** Raw payload and IP flows never leave local infrastructure.
- **Inference & Reconstruction Attacks:** Gradients and weight updates can inadvertently leak sensitive statistical characteristics or membership information (e.g., membership inference attacks, gradient inversion).
- **Future Protections (Phase 4+):** Differential Privacy (DP-SGD) to add calibrated noise, and Secure Aggregation (SecAgg) using cryptographic multi-party computation.
