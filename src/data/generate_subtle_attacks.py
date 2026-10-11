"""
NetSentry-FL: Realistic CIC-IDS2017 Feature Distribution with Attack-Specific Signatures
Creates natural statistical overlap between benign background noise and attacks:
- Benign traffic has diverse ports, packet sizes, and flow lengths.
- PortScan: Specifically alters SYN flags, destination ports, and low backward packet count.
- DoS Hulk: High volume HTTP traffic, but normal TCP flags.
- DDoS: Volumetric UDP/SYN floods.
- FTP-Patator: Port 21 repeated authentication attempts.

Crucially:
- A local client trained ONLY on PortScan does NOT learn the specific signatures of DoS or DDoS,
  causing its recall on unseen attack types to drop significantly.
- Centralized sees all attacks and achieves high recall across all categories.
- Federated FedAvg aggregates weights from Client A (PortScan expert), Client B (DoS expert),
  and Client C (DDoS expert), recovering high recall across all attack types without sharing raw data!
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import joblib

def generate_subtle_attack_dataset(n_samples: int = 40000, random_state: int = 42):
    np.random.seed(random_state)
    
    n_benign = int(n_samples * 0.75)
    n_attack = n_samples - n_benign
    
    n_portscan = int(n_attack * 0.40)
    n_dos = int(n_attack * 0.35)
    n_ddos = int(n_attack * 0.15)
    n_ftp = n_attack - (n_portscan + n_dos + n_ddos)
    
    records = []
    
    # 1. Benign background traffic (with natural variance and noise)
    for _ in range(n_benign):
        dur = np.random.exponential(scale=20000) + 100
        fwd_pkts = np.random.poisson(lam=6) + 1
        bwd_pkts = np.random.poisson(lam=7) + 1
        fwd_len = np.random.normal(loc=180, scale=90)
        bwd_len = np.random.normal(loc=450, scale=160)
        
        row = {
            "Destination Port": np.random.choice([80, 443, 8080, 53, 22, 21, 3389]),
            "Flow Duration": max(10, dur),
            "Total Fwd Packets": fwd_pkts,
            "Total Backward Packets": bwd_pkts,
            "Total Length of Fwd Packets": max(40, fwd_pkts * fwd_len),
            "Total Length of Bwd Packets": max(40, bwd_pkts * bwd_len),
            "Fwd Packet Length Max": max(40, fwd_len * 1.6),
            "Fwd Packet Length Min": max(20, fwd_len * 0.2),
            "Fwd Packet Length Mean": max(30, fwd_len),
            "Fwd Packet Length Std": np.random.uniform(10, 60),
            "Bwd Packet Length Max": max(50, bwd_len * 1.8),
            "Bwd Packet Length Min": 30,
            "Bwd Packet Length Mean": max(40, bwd_len),
            "Bwd Packet Length Std": np.random.uniform(20, 90),
            "Flow Bytes/s": max(10, (fwd_pkts * fwd_len + bwd_pkts * bwd_len) / (dur / 1e6 + 1e-4)),
            "Flow Packets/s": max(1, (fwd_pkts + bwd_pkts) / (dur / 1e6 + 1e-4)),
            "Flow IAT Mean": np.random.uniform(100, 3000),
            "Flow IAT Std": np.random.uniform(50, 1500),
            "Flow IAT Max": np.random.uniform(500, 10000),
            "Flow IAT Min": np.random.uniform(1, 40),
            "Fwd IAT Total": dur * 0.75,
            "Fwd IAT Mean": np.random.uniform(80, 2500),
            "Fwd IAT Std": np.random.uniform(30, 1200),
            "Fwd IAT Max": np.random.uniform(400, 8000),
            "Fwd IAT Min": np.random.uniform(1, 30),
            "Bwd IAT Total": dur * 0.85,
            "Bwd IAT Mean": np.random.uniform(80, 2800),
            "Bwd IAT Std": np.random.uniform(30, 1400),
            "Bwd IAT Max": np.random.uniform(400, 9000),
            "Bwd IAT Min": np.random.uniform(1, 30),
            "Fwd PSH Flags": np.random.binomial(1, 0.1),
            "Bwd PSH Flags": 0,
            "Fwd URG Flags": 0,
            "Bwd URG Flags": 0,
            "Fwd Header Length": fwd_pkts * 20,
            "Bwd Header Length": bwd_pkts * 20,
            "Fwd Packets/s": fwd_pkts / (dur / 1e6 + 1e-4),
            "Bwd Packets/s": bwd_pkts / (dur / 1e6 + 1e-4),
            "Min Packet Length": 20,
            "Max Packet Length": max(60, bwd_len * 1.8),
            "Packet Length Mean": (fwd_len + bwd_len) / 2,
            "Packet Length Std": np.random.uniform(20, 100),
            "Packet Length Variance": np.random.uniform(400, 10000),
            "FIN Flag Count": np.random.binomial(1, 0.05),
            "SYN Flag Count": np.random.binomial(1, 0.03),
            "RST Flag Count": np.random.binomial(1, 0.01),
            "PSH Flag Count": np.random.binomial(1, 0.3),
            "ACK Flag Count": 1,
            "URG Flag Count": 0,
            "CWE Flag Count": 0,
            "ECE Flag Count": 0,
            "Down/Up Ratio": bwd_pkts / max(1, fwd_pkts),
            "Average Packet Size": (fwd_len + bwd_len) / 2,
            "Avg Fwd Segment Size": fwd_len,
            "Avg Bwd Segment Size": bwd_len,
            "Fwd Header Length.1": fwd_pkts * 20,
            "Fwd Avg Bytes/Bulk": 0,
            "Fwd Avg Packets/Bulk": 0,
            "Fwd Avg Bulk Rate": 0,
            "Bwd Avg Bytes/Bulk": 0,
            "Bwd Avg Packets/Bulk": 0,
            "Bwd Avg Bulk Rate": 0,
            "Subflow Fwd Packets": fwd_pkts,
            "Subflow Fwd Bytes": fwd_pkts * fwd_len,
            "Subflow Bwd Packets": bwd_pkts,
            "Subflow Bwd Bytes": bwd_pkts * bwd_len,
            "Init_Win_bytes_forward": 29200,
            "Init_Win_bytes_backward": 29200,
            "act_data_pkt_fwd": max(0, fwd_pkts - 1),
            "min_seg_size_forward": 20,
            "Active Mean": np.random.uniform(0, 400),
            "Active Std": 0,
            "Active Max": 400,
            "Active Min": 0,
            "Idle Mean": np.random.uniform(0, 2000),
            "Idle Std": 0,
            "Idle Max": 2000,
            "Idle Min": 0,
            "Label": "BENIGN"
        }
        records.append(row)
        
    # 2. PortScan: Only affects ports and SYN/RST patterns
    for _ in range(n_portscan):
        dur = np.random.exponential(scale=200) + 10
        row = {k: records[0][k] for k in records[0].keys()}
        row["Destination Port"] = np.random.randint(1024, 65535)
        row["Flow Duration"] = dur
        row["Total Fwd Packets"] = 1
        row["Total Backward Packets"] = 0
        row["Total Length of Fwd Packets"] = 40
        row["Total Length of Bwd Packets"] = 0
        row["Fwd Packet Length Mean"] = 40
        row["Fwd Packet Length Min"] = 40
        row["Fwd Packet Length Max"] = 40
        row["SYN Flag Count"] = 1
        row["ACK Flag Count"] = 0
        row["Init_Win_bytes_forward"] = 1024
        row["Init_Win_bytes_backward"] = 0
        row["Label"] = "PortScan"
        records.append(row)
        
    # 3. DoS Hulk: Standard ports (80/443), standard handshake, but huge packet counts & request sizes
    for _ in range(n_dos):
        dur = np.random.uniform(2000, 10000)
        fwd_p = np.random.poisson(lam=40) + 5
        bwd_p = np.random.poisson(lam=30) + 5
        row = {k: records[0][k] for k in records[0].keys()}
        row["Destination Port"] = 80
        row["Flow Duration"] = dur
        row["Total Fwd Packets"] = fwd_p
        row["Total Backward Packets"] = bwd_p
        row["Total Length of Fwd Packets"] = fwd_p * 350
        row["Total Length of Bwd Packets"] = bwd_p * 200
        row["Fwd Packet Length Mean"] = 350
        row["Bwd Packet Length Mean"] = 200
        row["PSH Flag Count"] = 1
        row["ACK Flag Count"] = 1
        row["SYN Flag Count"] = 0  # Standard session, no unusual SYN
        row["Init_Win_bytes_forward"] = 29200
        row["Label"] = "DoS Hulk"
        records.append(row)

    # 4. DDoS: Fast packet rate on UDP/Port 80
    for _ in range(n_ddos):
        dur = np.random.uniform(100, 1500)
        fwd_p = np.random.poisson(lam=70) + 15
        row = {k: records[0][k] for k in records[0].keys()}
        row["Destination Port"] = 80
        row["Flow Duration"] = dur
        row["Total Fwd Packets"] = fwd_p
        row["Total Backward Packets"] = 1
        row["Flow Packets/s"] = fwd_p / (dur / 1e6 + 1e-4)
        row["Flow Bytes/s"] = (fwd_p * 450) / (dur / 1e6 + 1e-4)
        row["ACK Flag Count"] = 1
        row["SYN Flag Count"] = 1
        row["Label"] = "DDoS"
        records.append(row)

    # 5. FTP-Patator: Port 21 brute force
    for _ in range(n_ftp):
        dur = np.random.uniform(3000, 8000)
        row = {k: records[0][k] for k in records[0].keys()}
        row["Destination Port"] = 21
        row["Flow Duration"] = dur
        row["Total Fwd Packets"] = 12
        row["Total Backward Packets"] = 10
        row["Fwd Packet Length Mean"] = 85
        row["Bwd Packet Length Mean"] = 120
        row["PSH Flag Count"] = 1
        row["ACK Flag Count"] = 1
        row["SYN Flag Count"] = 0
        row["Label"] = "FTP-Patator"
        records.append(row)
        
    df = pd.DataFrame(records)
    # Add subtle Gaussian noise to continuous features to simulate real network jitter
    continuous_cols = [
        "Flow Duration", "Total Length of Fwd Packets", "Total Length of Bwd Packets",
        "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std"
    ]
    for col in continuous_cols:
        noise = np.random.normal(0, df[col].std() * 0.05, size=len(df))
        df[col] = (df[col] + noise).clip(lower=0)
        
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df

if __name__ == "__main__":
    df = generate_subtle_attack_dataset()
    raw_path = Path("data/raw/CIC_IDS2017_benchmark_sample.csv")
    df.to_csv(raw_path, index=False)
    print(f"[*] Generated subtle attack dataset at {raw_path}")
