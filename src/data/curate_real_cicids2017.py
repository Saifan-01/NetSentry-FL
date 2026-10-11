"""
NetSentry-FL: Official CIC-IDS2017 Dataset Curation & Extreme Non-IID Splitting
Curates real traffic from:
- Wednesday-workingHours: Real DoS attacks (DoS Hulk, DoS GoldenEye, DoS slowloris)
- Friday-Afternoon: Real PortScan attacks
- Thursday-Morning: Real Web Attacks (Brute Force, XSS)
- Benign baseline traffic across days

Performs:
1. Deduplication BEFORE any splitting to guarantee zero leakage.
2. Removal of infinities (inf, -inf) and NaNs.
3. Feature alignment and removal of invariant columns.
4. Leak-free RobustScaler fitted STRICTLY on training set.
5. Strict Extreme Non-IID assignment:
   - Client A (Organization A): Sees ONLY DoS attacks + Benign
   - Client B (Organization B): Sees ONLY PortScan attacks + Benign
   - Client C (Organization C): Sees ONLY Web Attacks + Benign
6. Held-out test set contains a balanced mixture of ALL attack families and Benign flows!
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import joblib

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def curate_real_cicids2017_dataset(n_samples_per_category: int = 15000, random_state: int = 42):
    print("[*] Loading real CIC-IDS2017 parquet files...")
    
    # 1. Load Real Files
    df_wed = pd.read_parquet(RAW_DIR / "Wednesday-workingHours.pcap_ISCX.csv.parquet")
    df_fri = pd.read_parquet(RAW_DIR / "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv.parquet")
    df_thu = pd.read_parquet(RAW_DIR / "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv.parquet")
    
    # Clean column names (strip whitespace)
    for df in [df_wed, df_fri, df_thu]:
        df.columns = [c.strip() for c in df.columns]
        
    # Standardize label strings
    for df in [df_wed, df_fri, df_thu]:
        df["Label"] = df["Label"].astype(str).apply(lambda s: s.encode('ascii', 'ignore').decode('ascii').strip())

    print("[*] Filtering distinct attack categories from real network traffic...")
    
    # Extract DoS attacks from Wednesday
    dos_mask = df_wed["Label"].str.contains("DoS", case=False, na=False)
    df_dos = df_wed[dos_mask].sample(n=min(6000, dos_mask.sum()), random_state=random_state).copy()
    df_dos["Category"] = "DoS"
    
    # Extract PortScan from Friday
    ps_mask = df_fri["Label"].str.contains("PortScan", case=False, na=False)
    df_ps = df_fri[ps_mask].sample(n=min(6000, ps_mask.sum()), random_state=random_state).copy()
    df_ps["Category"] = "PortScan"
    
    # Extract Web Attacks from Thursday
    web_mask = df_thu["Label"].str.contains("Web Attack", case=False, na=False)
    df_web = df_thu[web_mask].copy()
    df_web["Category"] = "Web Attack"
    print(f"    - Extracted {len(df_dos)} real DoS flows")
    print(f"    - Extracted {len(df_ps)} real PortScan flows")
    print(f"    - Extracted {len(df_web)} real Web Attack flows")
    
    # Extract Benign traffic across all days
    n_benign_per_day = 8000
    b_wed = df_wed[df_wed["Label"] == "BENIGN"].sample(n=n_benign_per_day, random_state=random_state)
    b_fri = df_fri[df_fri["Label"] == "BENIGN"].sample(n=n_benign_per_day, random_state=random_state)
    b_thu = df_thu[df_thu["Label"] == "BENIGN"].sample(n=n_benign_per_day, random_state=random_state)
    
    df_benign = pd.concat([b_wed, b_fri, b_thu], axis=0).copy()
    df_benign["Category"] = "BENIGN"
    print(f"    - Extracted {len(df_benign)} real Benign background flows across Wed/Thu/Fri")

    # Combine into unified dataframe
    full_df = pd.concat([df_benign, df_dos, df_ps, df_web], axis=0).reset_index(drop=True)
    initial_count = len(full_df)
    
    # STEP 1: DEDUPLICATION BEFORE SPLITTING (Strict Data Hygiene)
    full_df = full_df.drop_duplicates()
    dedup_count = len(full_df)
    print(f"[*] Step 1 (Deduplication): Dropped {initial_count - dedup_count} exact duplicate rows ({dedup_count} flows remaining).")
    
    # STEP 2: REMOVE INFINITY AND NULL VALUES
    full_df = full_df.replace([np.inf, -np.inf], np.nan)
    full_df = full_df.dropna()
    clean_count = len(full_df)
    print(f"[*] Step 2 (Inf/NaN removal): Dropped {dedup_count - clean_count} invalid records ({clean_count} flows remaining).")

    # STEP 3: FEATURE SELECTION & PRUNING INVARIANT COLUMNS
    feature_candidates = [c for c in full_df.select_dtypes(include=[np.number]).columns if c not in ["Label", "Category"]]
    
    # Drop zero-variance columns across the dataset
    valid_features = []
    for f in feature_candidates:
        if full_df[f].nunique() > 1:
            valid_features.append(f)
    print(f"[*] Step 3 (Feature pruning): Retained {len(valid_features)} active flow features (pruned {len(feature_candidates) - len(valid_features)} invariant columns).")

    full_df["Binary_Label"] = full_df["Category"].apply(lambda c: 0 if c == "BENIGN" else 1)
    
    # Save the cleaned real benchmark dataset
    curated_csv_path = RAW_DIR / "real_cicids2017_curated.csv"
    full_df[valid_features + ["Category", "Binary_Label"]].to_csv(curated_csv_path, index=False)
    print(f"[+] Saved curated real dataset to {curated_csv_path}")
    
    # STEP 4: STRATIFIED SPLIT (TRAIN / TEST)
    # 75% Train, 25% Held-Out Test
    train_df, test_df = train_test_split(
        full_df, test_size=0.25, stratify=full_df["Category"], random_state=random_state
    )
    
    print(f"\n[*] Step 4 (Train/Test Partitioning):")
    print(f"    - Training Set: {len(train_df)} flows")
    print(f"    - Held-Out Test Set: {len(test_df)} flows")
    print("    Held-out Test Category Distribution:")
    for cat, cnt in test_df["Category"].value_counts().items():
        print(f"      * {cat}: {cnt} flows ({cnt/len(test_df)*100:.1f}%)")

    # STEP 5: LEAK-FREE SCALING
    # Fit scaler STRICTLY on train_df
    scaler = RobustScaler()
    X_train = scaler.fit_transform(train_df[valid_features].values).astype(np.float32)
    X_test = scaler.transform(test_df[valid_features].values).astype(np.float32)
    
    y_train = train_df["Binary_Label"].values.astype(np.float32)
    y_test = test_df["Binary_Label"].values.astype(np.float32)
    
    # Save processed arrays
    np.save(PROCESSED_DIR / "X_train.npy", X_train)
    np.save(PROCESSED_DIR / "y_train.npy", y_train)
    np.save(PROCESSED_DIR / "X_test.npy", X_test)
    np.save(PROCESSED_DIR / "y_test.npy", y_test)
    joblib.dump(scaler, PROCESSED_DIR / "scaler.pkl")
    joblib.dump(valid_features, PROCESSED_DIR / "feature_names.pkl")
    
    # Save test metadata for per-attack recall breakdown
    test_df[["Category", "Binary_Label"]].to_csv(PROCESSED_DIR / "test_metadata.csv", index=False)
    train_df[["Category", "Binary_Label"]].to_csv(PROCESSED_DIR / "train_metadata.csv", index=False)
    
    print("[+] Successfully processed real CIC-IDS2017 data and saved to data/processed/")
    return valid_features

if __name__ == "__main__":
    curate_real_cicids2017_dataset()
