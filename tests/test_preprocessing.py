"""
Unit Tests for NetSentry-FL Data Preprocessing and Leakage Prevention
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import (
    clean_column_names,
    clean_infinite_and_nulls,
    drop_duplicates_safely,
    drop_constant_and_identifier_columns,
    split_and_scale_data
)

class TestDataPreprocessing(unittest.TestCase):
    
    def setUp(self):
        # Create small synthetic test dataframe with intentional anomalies
        np.random.seed(42)
        n = 100
        self.raw_df = pd.DataFrame({
            " Destination Port ": np.random.choice([80, 443, 22], n),
            "Flow Duration": np.random.exponential(1000, n),
            "Flow Bytes/s": np.random.normal(500, 50, n),
            "Constant_Col": [1.0] * n,
            "Source IP": [f"192.168.1.{i}" for i in range(n)],
            "Label": ["BENIGN"] * 70 + ["DoS"] * 30,
            "Category": ["BENIGN"] * 70 + ["DoS"] * 30
        })
        # Inject infinities and duplicates
        self.raw_df.loc[0, "Flow Bytes/s"] = np.inf
        self.raw_df.loc[1, "Flow Bytes/s"] = -np.inf
        # Duplicate row 5 at row 6
        self.raw_df.iloc[6] = self.raw_df.iloc[5]

    def test_clean_column_names(self):
        cleaned = clean_column_names(self.raw_df.copy())
        self.assertIn("Destination Port", cleaned.columns)
        self.assertNotIn(" Destination Port ", cleaned.columns)

    def test_clean_infinite_and_nulls(self):
        cleaned = clean_infinite_and_nulls(self.raw_df.copy())
        self.assertFalse(np.isinf(cleaned["Flow Bytes/s"]).any())
        self.assertFalse(cleaned["Flow Bytes/s"].isna().any())
        self.assertEqual(len(cleaned), 98)  # 2 rows with inf dropped

    def test_drop_duplicates_safely(self):
        cleaned = drop_duplicates_safely(self.raw_df.copy())
        self.assertEqual(len(cleaned), 99)  # 1 duplicate dropped

    def test_deterministic_drop_constant_and_identifiers(self):
        df_clean_cols = clean_column_names(self.raw_df.copy())
        df_pruned, dropped = drop_constant_and_identifier_columns(df_clean_cols)
        
        # Verify deterministic alphabetical sorting
        self.assertEqual(dropped, sorted(dropped))
        self.assertIn("Constant_Col", dropped)
        self.assertIn("Source IP", dropped)
        self.assertNotIn("Constant_Col", df_pruned.columns)
        self.assertNotIn("Source IP", df_pruned.columns)

    def test_leak_free_70_10_20_stratified_split(self):
        df = clean_column_names(self.raw_df.copy())
        df = clean_infinite_and_nulls(df)
        df = drop_duplicates_safely(df)
        df["Binary_Label"] = df["Category"].apply(lambda c: 0 if c == "BENIGN" else 1)
        
        feature_cols = ["Destination Port", "Flow Duration", "Flow Bytes/s"]
        splits = split_and_scale_data(
            df=df,
            feature_cols=feature_cols,
            target_col="Binary_Label",
            stratify_col="Category",
            test_size=0.20,
            val_size=0.10,
            random_state=42
        )
        
        train_idx = set(splits["df_train"].index)
        val_idx = set(splits["df_val"].index)
        test_idx = set(splits["df_test"].index)
        
        # Ensure zero overlap
        self.assertEqual(len(train_idx.intersection(val_idx)), 0)
        self.assertEqual(len(train_idx.intersection(test_idx)), 0)
        self.assertEqual(len(val_idx.intersection(test_idx)), 0)
        
        # Verify shapes
        total = len(df)
        self.assertEqual(len(splits["X_train"]) + len(splits["X_val"]) + len(splits["X_test"]), total)
        self.assertTrue(0.68 <= len(splits["X_train"]) / total <= 0.72)
        self.assertTrue(0.08 <= len(splits["X_val"]) / total <= 0.12)
        self.assertTrue(0.18 <= len(splits["X_test"]) / total <= 0.22)

if __name__ == "__main__":
    unittest.main()
