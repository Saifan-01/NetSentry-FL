"""
NetSentry-FL: Core Data Preprocessing and Ingestion Module
Handles cleaning, missing/infinite value handling, heavy-tail transformation,
normalization, and stratified dataset splitting for CIC-IDS2017 flow data.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, RobustScaler
import joblib

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strips leading and trailing whitespace from column names."""
    df.columns = [c.strip() for c in df.columns]
    return df

def clean_infinite_and_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces +/- Inf with NaN, then drops rows containing missing entries.
    In CIC-IDS2017, Flow Bytes/s and Flow Packets/s frequently have 'Infinity'.
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    initial_len = len(df)
    df = df.dropna()
    dropped = initial_len - len(df)
    if dropped > 0:
        print(f"[*] Dropped {dropped} rows containing NaN or Infinite values ({dropped / initial_len * 100:.2f}%).")
    return df

def drop_duplicates_safely(df: pd.DataFrame) -> pd.DataFrame:
    """Removes duplicate flow records before dataset splitting to guarantee zero leakage."""
    initial_len = len(df)
    df = df.drop_duplicates()
    dropped = initial_len - len(df)
    if dropped > 0:
        print(f"[*] Dropped {dropped} duplicate flow records ({dropped / initial_len * 100:.2f}%).")
    return df

def map_binary_labels(df: pd.DataFrame, label_col: str = "Label") -> pd.DataFrame:
    """
    Converts multi-class attack strings into binary labels:
    0 = BENIGN
    1 = ATTACK
    Preserves original string label in 'Category' or 'Label'.
    """
    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found in dataframe.")
    
    df[label_col] = df[label_col].astype(str).str.strip()
    df["Binary_Label"] = df[label_col].apply(lambda x: 0 if x.upper() == "BENIGN" else 1)
    return df

def drop_constant_and_identifier_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Deterministically identifies and drops non-generalizable identifier columns
    (e.g., Flow ID, Source IP) and zero-variance constant numeric columns.
    Uses sorted set operations and explicit column dropping without silent error suppression.
    """
    cols_to_drop_set = set()
    
    # 1. Target known identifier columns
    identifiers = ["Flow ID", "Source IP", "Destination IP", "Timestamp", "External IP"]
    for identifier in identifiers:
        if identifier in df.columns:
            cols_to_drop_set.add(identifier)
            
    # 2. Check numeric columns for zero variance (single unique value)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ["Label", "Binary_Label", "Category"] and df[col].nunique() <= 1:
            cols_to_drop_set.add(col)
            
    cols_to_drop = sorted(cols_to_drop_set)
    existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df_cleaned = df.drop(columns=existing_cols_to_drop)
    
    print(f"[*] Deterministically dropped {len(existing_cols_to_drop)} invariant/identifier columns: {existing_cols_to_drop[:5]}...")
    return df_cleaned, existing_cols_to_drop

def split_and_scale_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "Binary_Label",
    stratify_col: str = "Category",
    test_size: float = 0.20,
    val_size: float = 0.10,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Splits data into 70% Train, 10% Validation, and 20% Held-Out Test using Stratified sampling.
    Applies QuantileTransformer (Gaussian output mapped to [-5, 5]) fitted STRICTLY on training
    data to handle extreme heavy-tailed network features and eliminate outlier gradient explosion.
    """
    stratify_target = df[stratify_col] if stratify_col in df.columns else df[target_col]
    
    # Step 1: Split out test set (20% held-out test)
    df_train_val, df_test = train_test_split(
        df,
        test_size=test_size,
        stratify=stratify_target,
        random_state=random_state
    )
    
    # Step 2: Split train_val into 70% train and 10% val (10 / 80 = 0.125)
    val_ratio = val_size / (1.0 - test_size)
    stratify_train_val = df_train_val[stratify_col] if stratify_col in df_train_val.columns else df_train_val[target_col]
    
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=val_ratio,
        stratify=stratify_train_val,
        random_state=random_state
    )
    
    # Extract feature arrays
    X_train_raw = df_train[feature_cols].values.astype(np.float64)
    X_val_raw = df_val[feature_cols].values.astype(np.float64)
    X_test_raw = df_test[feature_cols].values.astype(np.float64)
    
    y_train = df_train[target_col].values.astype(np.float32)
    y_val = df_val[target_col].values.astype(np.float32)
    y_test = df_test[target_col].values.astype(np.float32)
    
    # Step 3: Fit QuantileTransformer strictly on training split
    # Maps heavy-tailed distribution to standard normal N(0, 1) and clips to [-5, 5]
    scaler = QuantileTransformer(output_distribution="normal", random_state=random_state, n_quantiles=min(1000, len(X_train_raw)))
    X_train_scaled = np.clip(scaler.fit_transform(X_train_raw), -5.0, 5.0).astype(np.float32)
    X_val_scaled = np.clip(scaler.transform(X_val_raw), -5.0, 5.0).astype(np.float32)
    X_test_scaled = np.clip(scaler.transform(X_test_raw), -5.0, 5.0).astype(np.float32)
    
    return {
        "X_train": X_train_scaled,
        "y_train": y_train,
        "X_val": X_val_scaled,
        "y_val": y_val,
        "X_test": X_test_scaled,
        "y_test": y_test,
        "df_train": df_train,
        "df_val": df_val,
        "df_test": df_test,
        "scaler": scaler,
        "feature_names": feature_cols
    }
