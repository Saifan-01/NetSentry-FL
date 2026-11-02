"""
Generate high-fidelity sample CSVs from the curated dataset for the Streamlit demo.
"""
import pandas as pd
from pathlib import Path
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CURATED_CSV = PROJECT_ROOT / "data" / "raw" / "real_cicids2017_curated.csv"
DEMO_DIR = PROJECT_ROOT / "data" / "demo_samples"
DEMO_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CURATED_CSV)
feature_names = joblib.load(PROJECT_ROOT / "data" / "processed" / "feature_names.pkl")

# 1. Mixed Enterprise Traffic (100 flows: 70 Benign, 15 DoS, 10 PortScan, 5 Web Attack)
sample_benign = df[df["Category"] == "BENIGN"].sample(70, random_state=42)
sample_dos = df[df["Category"] == "DoS"].sample(15, random_state=42)
sample_ps = df[df["Category"] == "PortScan"].sample(10, random_state=42)
sample_web = df[df["Category"] == "Web Attack"].sample(5, random_state=42)

demo_mixed = pd.concat([sample_benign, sample_dos, sample_ps, sample_web]).sample(frac=1.0, random_state=42).reset_index(drop=True)
demo_mixed.to_csv(DEMO_DIR / "demo_mixed_enterprise_traffic.csv", index=False)

# 2. Active Cyberattack Incident (50 flows: 35 Attacks, 15 Benign)
sample_attack_heavy = pd.concat([
    df[df["Category"] == "DoS"].sample(15, random_state=99),
    df[df["Category"] == "PortScan"].sample(15, random_state=99),
    df[df["Category"] == "Web Attack"].sample(5, random_state=99),
    df[df["Category"] == "BENIGN"].sample(15, random_state=99)
]).sample(frac=1.0, random_state=99).reset_index(drop=True)
sample_attack_heavy.to_csv(DEMO_DIR / "demo_attack_incident_traffic.csv", index=False)

# 3. Clean Regular Business Traffic (50 flows: 100% Benign)
sample_clean = df[df["Category"] == "BENIGN"].sample(50, random_state=123).reset_index(drop=True)
sample_clean.to_csv(DEMO_DIR / "demo_clean_business_traffic.csv", index=False)

print("[+] Successfully generated demo CSV samples matching 70/10/20 test split features.")
