"""
NetSentry-FL: Official CIC-IDS2017 Dataset Curation & 70/10/20 Stratified Splitting
Curates real network traffic from:
- Wednesday-workingHours: Real DoS attacks (DoS Hulk, DoS GoldenEye, DoS slowloris)
- Friday-Afternoon: Real PortScan attacks
- Thursday-Morning: Real Web Attacks (Brute Force, XSS)
- Multi-day Benign baseline traffic

Performs:
1. Column name whitespace cleanup.
2. Deduplication BEFORE any splitting to guarantee zero leakage.
3. Removal of infinities (inf, -inf) and NaNs.
4. Deterministic pruning of invariant and identifier columns.
5. Strict 70% Train, 10% Validation, 20% Held-Out Test stratified split.
6. RobustScaler fitted STRICTLY on the 70% training set only.
7. Saves all splits, scaler, feature list, and metadata for per-attack evaluation.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib

from src.data.preprocess import (
    clean_column_names,
    clean_infinite_and_nulls,
    drop_duplicates_safely,
    drop_constant_and_identifier_columns,
    split_and_scale_data
)

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def curate_real_cicids2017(random_state: int = 42) -> List[str]:
    print("[*] Loading real CIC-IDS2017 raw parquet files...")
    
    wed_path = RAW_DIR / "Wednesday-workingHours.pcap_ISCX.csv.parquet"
    fri_path = RAW_DIR / "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv.parquet"
    thu_path = RAW_DIR / "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv.parquet"
    
    if not (wed_path.exists() and fri_path.exists() and thu_path.exists()):
        raise FileNotFoundError(f"Missing required raw parquet files in {RAW_DIR}")
        
    df_wed = pd.read_parquet(wed_path)
    df_fri = pd.read_parquet(fri_path)
    df_thu = pd.read_parquet(thu_path)
    
    # Step 1: Clean column names (strip whitespace)
    for df in [df_wed, df_fri, df_thu]:
        clean_column_names(df)
        df["Label"] = df["Label"].astype(str).apply(lambda s: s.encode('ascii', 'ignore').decode('ascii').strip())
        
    print("[*] Extracting distinct attack categories from real network traffic...")
    
    # 1. DoS attacks from Wednesday
    dos_mask = df_wed["Label"].str.contains("DoS", case=False, na=False)
    df_dos = df_wed[dos_mask].sample(n=min(6000, dos_mask.sum()), random_state=random_state).copy()
    df_dos["Category"] = "DoS"
    
    # 2. PortScan from Friday
    ps_mask = df_fri["Label"].str.contains("PortScan", case=False, na=False)
    df_ps = df_fri[ps_mask].sample(n=min(6000, ps_mask.sum()), random_state=random_state).copy()
    df_ps["Category"] = "PortScan"
    
    # 3. Web Attacks from Thursday
    web_mask = df_thu["Label"].str.contains("Web Attack", case=False, na=False)
    df_web = df_thu[web_mask].copy()
    df_web["Category"] = "Web Attack"
    
    # 4. Multi-day Benign traffic
    n_benign_per_day = 8000
    b_wed = df_wed[df_wed["Label"] == "BENIGN"].sample(n=n_benign_per_day, random_state=random_state)
    b_fri = df_fri[df_fri["Label"] == "BENIGN"].sample(n=n_benign_per_day, random_state=random_state)
    b_thu = df_thu[df_thu["Label"] == "BENIGN"].sample(n=n_benign_per_day, random_state=random_state)
    
    df_benign = pd.concat([b_wed, b_fri, b_thu], axis=0).copy()
    df_benign["Category"] = "BENIGN"
    
    # Combine into unified dataset
    full_df = pd.concat([df_benign, df_dos, df_ps, df_web], axis=0).reset_index(drop=True)
    print(f"[*] Combined total raw records: {len(full_df)}")
    
    # Step 2: Deduplication before splitting (Strict Data Hygiene)
    full_df = drop_duplicates_safely(full_df)
    
    # Step 3: Clean infinite and null values
    full_df = clean_infinite_and_nulls(full_df)
    
    # Step 4: Map binary label while keeping original category
    full_df["Binary_Label"] = full_df["Category"].apply(lambda c: 0 if c == "BENIGN" else 1)
    
    # Step 5: Deterministic dropping of constant & identifier columns
    full_df, dropped_cols = drop_constant_and_identifier_columns(full_df)
    
    # Feature columns are numeric columns excluding labels and metadata
    feature_cols = sorted([c for c in full_df.select_dtypes(include=[np.number]).columns if c not in ["Binary_Label"]])
    print(f"[*] Retained {len(feature_cols)} clean predictive flow features.")
    
    # Step 6: 70% Train, 10% Validation, 20% Held-Out Test Stratified Split
    print("[*] Performing 70/10/20 stratified split...")
    splits = split_and_scale_data(
        df=full_df,
        feature_cols=feature_cols,
        target_col="Binary_Label",
        stratify_col="Category",
        test_size=0.20,
        val_size=0.10,
        random_state=random_state
    )
    
    # Step 7: Leakage verification check
    train_indices = set(splits["df_train"].index)
    val_indices = set(splits["df_val"].index)
    test_indices = set(splits["df_test"].index)
    
    overlap_train_val = train_indices.intersection(val_indices)
    overlap_train_test = train_indices.intersection(test_indices)
    overlap_val_test = val_indices.intersection(test_indices)
    
    assert len(overlap_train_val) == 0, f"Leakage detected: {len(overlap_train_val)} overlapping rows between train and val!"
    assert len(overlap_train_test) == 0, f"Leakage detected: {len(overlap_train_test)} overlapping rows between train and test!"
    assert len(overlap_val_test) == 0, f"Leakage detected: {len(overlap_val_test)} overlapping rows between val and test!"
    print("[+] Verified ZERO index/row overlap across train, val, and test splits!")
    
    # Step 8: Save processed arrays, scaler, and metadata
    np.save(PROCESSED_DIR / "X_train.npy", splits["X_train"])
    np.save(PROCESSED_DIR / "y_train.npy", splits["y_train"])
    np.save(PROCESSED_DIR / "X_val.npy", splits["X_val"])
    np.save(PROCESSED_DIR / "y_val.npy", splits["y_val"])
    np.save(PROCESSED_DIR / "X_test.npy", splits["X_test"])
    np.save(PROCESSED_DIR / "y_test.npy", splits["y_test"])
    
    joblib.dump(splits["scaler"], PROCESSED_DIR / "scaler.pkl")
    joblib.dump(feature_cols, PROCESSED_DIR / "feature_names.pkl")
    
    # Save metadata for per-attack analysis
    splits["df_train"][["Category", "Binary_Label"]].to_csv(PROCESSED_DIR / "train_metadata.csv", index=False)
    splits["df_val"][["Category", "Binary_Label"]].to_csv(PROCESSED_DIR / "val_metadata.csv", index=False)
    splits["df_test"][["Category", "Binary_Label"]].to_csv(PROCESSED_DIR / "test_metadata.csv", index=False)
    
    # Save curated full dataset as a benchmark reference
    full_df[feature_cols + ["Category", "Binary_Label"]].to_csv(RAW_DIR / "real_cicids2017_curated.csv", index=False)
    
    print("\n[+] Dataset Curation Summary:")
    print(f"    - Train split (70%): {splits['X_train'].shape} (Benign: {(splits['y_train']==0).sum()}, Attack: {(splits['y_train']==1).sum()})")
    print(f"    - Val split   (10%): {splits['X_val'].shape} (Benign: {(splits['y_val']==0).sum()}, Attack: {(splits['y_val']==1).sum()})")
    print(f"    - Test split  (20%): {splits['X_test'].shape} (Benign: {(splits['y_test']==0).sum()}, Attack: {(splits['y_test']==1).sum()})")
    print("    Test Category Distribution:")
    for cat, count in splits["df_test"]["Category"].value_counts().items():
        print(f"      * {cat}: {count} flows ({count/len(splits['df_test'])*100:.1f}%)")
        
    return feature_cols

if __name__ == "__main__":
    curate_real_cicids2017()
