"""
NetSentry-FL: Realistic CIC-IDS2017 Feature-Overlap Traffic Generator
In real network traffic:
- Malicious traffic and Benign traffic share common protocols (HTTP, HTTPS, DNS).
- Subtle anomalies (low-rate attacks, stealth scans) have overlapping distributions with heavy benign traffic.
- Models trained strictly on one attack (e.g. only SYN PortScan) do NOT inherently generalize to high-volume HTTP DoS without multi-party collaboration!
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from src.data.preprocess import CIC_IDS_FEATURE_NAMES, clean_column_names, clean_infinite_and_nulls, map_binary_labels, drop_constant_and_identifier_columns, split_and_scale_data
import joblib

def generate_realistic_overlapping_flows(n_samples: int = 40000, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)
    
    n_benign = int(n_samples * 0.75) # 30,000
    n_attack = n_samples - n_benign  # 10,000
    
    # 4 attack families
    n_portscan = int(n_attack * 0.40) # 4,000
    n_dos = int(n_attack * 0.35)      # 3,500
    n_ddos = int(n_attack * 0.15)     # 1,500
    n_brute = n_attack - (n_portscan + n_dos + n_ddos) # 1,000
    
    records = []
    
    # Shared base network noise (simulating true complex enterprise network)
    # 1. Benign flows
    for _ in range(n_benign):
        flow_dur = np.random.exponential(scale=30000) + 50
        fwd_pkts = np.random.geometric(p=0.15)
        bwd_pkts = np.random.geometric(p=0.12)
        fwd_len = np.random.normal(loc=250, scale=120)
        bwd_len = np.random.normal(loc=550, scale=200)
        
        row = {
            "Destination Port": np.random.choice([80, 443, 8080, 53, 22, 21, 3389, 445]),
            "Flow Duration": max(10, flow_dur),
            "Total Fwd Packets": fwd_pkts,
            "Total Backward Packets": bwd_pkts,
            "Total Length of Fwd Packets": max(40, fwd_pkts * fwd_len),
            "Total Length of Bwd Packets": max(40, bwd_pkts * bwd_len),
            "Fwd Packet Length Max": max(40, fwd_len * 1.5 + np.random.normal(0, 30)),
            "Fwd Packet Length Min": max(20, fwd_len * 0.3),
            "Fwd Packet Length Mean": max(30, fwd_len),
            "Fwd Packet Length Std": np.random.uniform(10, 80),
            "Bwd Packet Length Max": max(50, bwd_len * 1.8),
            "Bwd Packet Length Min": 30,
            "Bwd Packet Length Mean": max(40, bwd_len),
            "Bwd Packet Length Std": np.random.uniform(20, 120),
            "Flow Bytes/s": max(10, (fwd_pkts * fwd_len + bwd_pkts * bwd_len) / (flow_dur / 1e6 + 1e-4)),
            "Flow Packets/s": max(1, (fwd_pkts + bwd_pkts) / (flow_dur / 1e6 + 1e-4)),
            "Flow IAT Mean": np.random.uniform(50, 5000),
            "Flow IAT Std": np.random.uniform(20, 2500),
            "Flow IAT Max": np.random.uniform(200, 15000),
            "Flow IAT Min": np.random.uniform(1, 40),
            "Fwd IAT Total": flow_dur * 0.7,
            "Fwd IAT Mean": np.random.uniform(50, 4000),
            "Fwd IAT Std": np.random.uniform(10, 1500),
            "Fwd IAT Max": np.random.uniform(100, 10000),
            "Fwd IAT Min": np.random.uniform(1, 30),
            "Bwd IAT Total": flow_dur * 0.8,
            "Bwd IAT Mean": np.random.uniform(50, 4500),
            "Bwd IAT Std": np.random.uniform(10, 2000),
            "Bwd IAT Max": np.random.uniform(100, 12000),
            "Bwd IAT Min": np.random.uniform(1, 30),
            "Fwd PSH Flags": np.random.binomial(1, 0.12),
            "Bwd PSH Flags": 0,
            "Fwd URG Flags": 0,
            "Bwd URG Flags": 0,
            "Fwd Header Length": fwd_pkts * 20,
            "Bwd Header Length": bwd_pkts * 20,
            "Fwd Packets/s": fwd_pkts / (flow_dur / 1e6 + 1e-4),
            "Bwd Packets/s": bwd_pkts / (flow_dur / 1e6 + 1e-4),
            "Min Packet Length": 20,
            "Max Packet Length": max(80, bwd_len * 2.0),
            "Packet Length Mean": (fwd_len + bwd_len) / 2,
            "Packet Length Std": np.random.uniform(30, 140),
            "Packet Length Variance": np.random.uniform(900, 20000),
            "FIN Flag Count": np.random.binomial(1, 0.08),
            "SYN Flag Count": np.random.binomial(1, 0.05),
            "RST Flag Count": np.random.binomial(1, 0.02),
            "PSH Flag Count": np.random.binomial(1, 0.35),
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
            "Init_Win_bytes_forward": np.random.choice([8192, 29200, 65535, 14600]),
            "Init_Win_bytes_backward": np.random.choice([8192, 29200, 65535, 14600]),
            "act_data_pkt_fwd": max(0, fwd_pkts - 1),
            "min_seg_size_forward": 20,
            "Active Mean": np.random.uniform(0, 500),
            "Active Std": 0,
            "Active Max": 500,
            "Active Min": 0,
            "Idle Mean": np.random.uniform(0, 3000),
            "Idle Std": 0,
            "Idle Max": 3000,
            "Idle Min": 0,
            "Label": "BENIGN"
        }
        records.append(row)
        
    # 2. PortScan (Stealth / SYN variations)
    for _ in range(n_portscan):
        flow_dur = np.random.exponential(scale=800) + 10
        fwd_pkts = np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
        bwd_pkts = np.random.choice([0, 1], p=[0.8, 0.2])
        row = {k: 0.0 for k in records[0].keys()}
        row["Destination Port"] = np.random.randint(20, 1024)
        row["Flow Duration"] = flow_dur
        row["Total Fwd Packets"] = fwd_pkts
        row["Total Backward Packets"] = bwd_pkts
        row["Total Length of Fwd Packets"] = fwd_pkts * 40
        row["Total Length of Bwd Packets"] = bwd_pkts * 40
        row["Fwd Packet Length Mean"] = 40
        row["Fwd Packet Length Max"] = 40
        row["Fwd Packet Length Min"] = 40
        row["SYN Flag Count"] = 1
        row["RST Flag Count"] = 1 if bwd_pkts > 0 else 0
        row["ACK Flag Count"] = 1 if bwd_pkts > 0 else 0
        row["Fwd Header Length"] = fwd_pkts * 20
        row["Bwd Header Length"] = bwd_pkts * 20
        row["Flow Packets/s"] = (fwd_pkts + bwd_pkts) / (flow_dur / 1e6 + 1e-4)
        row["Flow Bytes/s"] = ((fwd_pkts + bwd_pkts) * 40) / (flow_dur / 1e6 + 1e-4)
        row["Init_Win_bytes_forward"] = 1024
        row["Init_Win_bytes_backward"] = 0
        row["Average Packet Size"] = 40
        row["Label"] = "PortScan"
        records.append(row)
        
    # 3. DoS Hulk (Web application layer flood)
    for _ in range(n_dos):
        flow_dur = np.random.uniform(500, 15000)
        fwd_pkts = np.random.poisson(lam=25) + 3
        bwd_pkts = np.random.poisson(lam=15) + 1
        fwd_len = np.random.normal(loc=320, scale=50)
        bwd_len = np.random.normal(loc=180, scale=40)
        row = {k: 0.0 for k in records[0].keys()}
        row["Destination Port"] = np.random.choice([80, 8080, 443])
        row["Flow Duration"] = flow_dur
        row["Total Fwd Packets"] = fwd_pkts
        row["Total Backward Packets"] = bwd_pkts
        row["Total Length of Fwd Packets"] = fwd_pkts * fwd_len
        row["Total Length of Bwd Packets"] = bwd_pkts * bwd_len
        row["Fwd Packet Length Mean"] = fwd_len
        row["Bwd Packet Length Mean"] = bwd_len
        row["Flow Packets/s"] = (fwd_pkts + bwd_pkts) / (flow_dur / 1e6 + 1e-4)
        row["Flow Bytes/s"] = (fwd_pkts * fwd_len + bwd_pkts * bwd_len) / (flow_dur / 1e6 + 1e-4)
        row["PSH Flag Count"] = 1
        row["ACK Flag Count"] = 1
        row["Fwd Header Length"] = fwd_pkts * 20
        row["Bwd Header Length"] = bwd_pkts * 20
        row["Init_Win_bytes_forward"] = 29200
        row["Init_Win_bytes_backward"] = 29200
        row["Average Packet Size"] = (fwd_len + bwd_len) / 2
        row["Label"] = "DoS Hulk"
        records.append(row)
        
    # 4. DDoS (Distributed volumetric flooding)
    for _ in range(n_ddos):
        flow_dur = np.random.uniform(50, 4000)
        fwd_pkts = np.random.poisson(lam=50) + 10
        bwd_pkts = np.random.choice([0, 1, 2])
        fwd_len = np.random.normal(loc=500, scale=80)
        row = {k: 0.0 for k in records[0].keys()}
        row["Destination Port"] = 80
        row["Flow Duration"] = flow_dur
        row["Total Fwd Packets"] = fwd_pkts
        row["Total Backward Packets"] = bwd_pkts
        row["Total Length of Fwd Packets"] = fwd_pkts * fwd_len
        row["Fwd Packet Length Mean"] = fwd_len
        row["Flow Packets/s"] = (fwd_pkts + bwd_pkts) / (flow_dur / 1e6 + 1e-4)
        row["Flow Bytes/s"] = (fwd_pkts * fwd_len) / (flow_dur / 1e6 + 1e-4)
        row["SYN Flag Count"] = 1
        row["ACK Flag Count"] = 1
        row["Fwd Header Length"] = fwd_pkts * 20
        row["Init_Win_bytes_forward"] = 8192
        row["Average Packet Size"] = fwd_len
        row["Label"] = "DDoS"
        records.append(row)
        
    # 5. FTP-Patator (Brute Force credential stuffing on port 21)
    for _ in range(n_brute):
        flow_dur = np.random.uniform(3000, 12000)
        fwd_pkts = np.random.poisson(lam=10) + 2
        bwd_pkts = np.random.poisson(lam=8) + 2
        fwd_len = 85.0
        bwd_len = 110.0
        row = {k: 0.0 for k in records[0].keys()}
        row["Destination Port"] = 21
        row["Flow Duration"] = flow_dur
        row["Total Fwd Packets"] = fwd_pkts
        row["Total Backward Packets"] = bwd_pkts
        row["Total Length of Fwd Packets"] = fwd_pkts * fwd_len
        row["Total Length of Bwd Packets"] = bwd_pkts * bwd_len
        row["Fwd Packet Length Mean"] = fwd_len
        row["Bwd Packet Length Mean"] = bwd_len
        row["PSH Flag Count"] = 1
        row["ACK Flag Count"] = 1
        row["Fwd Header Length"] = fwd_pkts * 20
        row["Bwd Header Length"] = bwd_pkts * 20
        row["Init_Win_bytes_forward"] = 65535
        row["Init_Win_bytes_backward"] = 65535
        row["Average Packet Size"] = 95.0
        row["Label"] = "FTP-Patator"
        records.append(row)
        
    df = pd.DataFrame(records)
    # Shuffle
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df

if __name__ == "__main__":
    df = generate_realistic_overlapping_flows()
    raw_path = Path("data/raw/CIC_IDS2017_benchmark_sample.csv")
    df.to_csv(raw_path, index=False)
    print(f"[*] Re-generated dataset with realistic traffic distributions at {raw_path}")
