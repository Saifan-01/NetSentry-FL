"""
NetSentry-FL: Data Preprocessing and Ingestion Module
Handles cleaning, missing/infinite value handling, normalization,
and stratified dataset splitting for CIC-IDS2017 flow data.
"""

import os
import re
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib

CIC_IDS_FEATURE_NAMES = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length",
    "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance", "FIN Flag Count",
    "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size",
    "Avg Bwd Segment Size", "Fwd Header Length.1", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min"
]

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strips leading/trailing whitespaces and regularizes column names."""
    df.columns = [c.strip() for c in df.columns]
    return df

def clean_infinite_and_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces +/- Inf with NaN, then drops or imputes missing entries.
    In CIC-IDS2017, Flow Bytes/s and Flow Packets/s frequently have 'Infinity'.
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    initial_len = len(df)
    df = df.dropna()
    dropped = initial_len - len(df)
    if dropped > 0:
        print(f"[*] Dropped {dropped} rows containing NaN or Infinite values ({dropped / initial_len * 100:.2f}%).")
    return df

def map_binary_labels(df: pd.DataFrame, label_col: str = "Label") -> pd.DataFrame:
    """
    Converts multi-class attack strings into binary labels:
    0 = BENIGN
    1 = ATTACK
    """
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataframe.")
    
    # Standardize string values
    df[label_col] = df[label_col].astype(str).str.strip()
    df["Binary_Label"] = df[label_col].apply(lambda x: 0 if x.upper() == "BENIGN" else 1)
    return df

def drop_constant_and_identifier_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Removes non-generalizable columns like Flow ID, Source IP, Timestamp,
    and columns with zero variance (constant values across all rows).
    """
    cols_to_drop = []
    # Drop identifiers if present
    for identifier in ["Flow ID", "Source IP", "Destination IP", "Timestamp", "External IP"]:
        if identifier in df.columns:
            cols_to_drop.append(identifier)
            
    # Check numeric columns for zero standard deviation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ["Label", "Binary_Label"] and df[col].nunique() <= 1:
            cols_to_drop.append(col)
            
    cols_to_drop = list(set(cols_to_drop))
    df_cleaned = df.drop(columns=cols_to_drop, errors="ignore")
    print(f"[*] Dropped {len(cols_to_drop)} invariant/identifier columns: {cols_to_drop[:5]}...")
    return df_cleaned, cols_to_drop

def split_and_scale_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "Binary_Label",
    test_size: float = 0.20,
    val_size: float = 0.10,
    random_state: int = 42
) -> Dict[str, np.ndarray]:
    """
    Splits data into train, val, test using Stratified sampling,
    and applies RobustScaler/StandardScaler fitted ONLY on training data to prevent leakage.
    """
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Step 1: Split out test set (held-out)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # Step 2: Split train and validation
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, stratify=y_train_val, random_state=random_state
    )
    
    # Step 3: Fit scaler strictly on X_train to prevent data leakage
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        "X_train": X_train_scaled,
        "y_train": y_train,
        "X_val": X_val_scaled,
        "y_val": y_val,
        "X_test": X_test_scaled,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": feature_cols
    }
