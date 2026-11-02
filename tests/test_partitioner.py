"""
Unit Tests for NetSentry-FL Extreme Non-IID Partitioner
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.evaluate_real_cicids import create_extreme_non_iid_silos

class TestPartitioner(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(42)
        n = 300
        # 180 Benign, 40 DoS, 40 PortScan, 40 Web Attack
        categories = ["BENIGN"] * 180 + ["DoS"] * 40 + ["PortScan"] * 40 + ["Web Attack"] * 40
        self.train_meta = pd.DataFrame({
            "Category": categories,
            "Binary_Label": [0 if c == "BENIGN" else 1 for c in categories]
        })
        self.X_train = np.random.randn(n, 10).astype(np.float32)

    def test_extreme_non_iid_silos(self):
        silos = create_extreme_non_iid_silos(self.X_train, self.train_meta, seed=42)
        
        # Verify 3 clients are created
        self.assertEqual(len(silos), 3)
        self.assertIn(0, silos)
        self.assertIn(1, silos)
        self.assertIn(2, silos)
        
        # Client 0 should have 60 Benign + 40 DoS = 100 flows
        self.assertEqual(len(silos[0]["X"]), 100)
        self.assertEqual(len(silos[0]["y"]), 100)
        self.assertEqual((silos[0]["y"] == 1).sum(), 40)
        
        # Client 1 should have 60 Benign + 40 PortScan = 100 flows
        self.assertEqual(len(silos[1]["X"]), 100)
        self.assertEqual(len(silos[1]["y"]), 100)
        self.assertEqual((silos[1]["y"] == 1).sum(), 40)
        
        # Client 2 should have 60 Benign + 40 Web Attack = 100 flows
        self.assertEqual(len(silos[2]["X"]), 100)
        self.assertEqual(len(silos[2]["y"]), 100)
        self.assertEqual((silos[2]["y"] == 1).sum(), 40)

if __name__ == "__main__":
    unittest.main()
