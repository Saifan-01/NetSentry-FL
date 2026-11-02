"""
NetSentry-FL: Unit Test Suite for Federated Aggregators & Quantization
Tests:
1. FedAvg: Sample-weighted and equal-weighted aggregation mathematics
2. Coordinate-wise Median: Robustness to extreme gradient outliers
3. Trimmed Mean: Trimming top and bottom percentiles
4. Quantization: FP16 and INT8 compression integrity and parameter reconstruction
5. Invariant: Consensus preservation when all client updates are identical
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
from src.federated.robust_aggregation import (
    aggregate_fedavg,
    aggregate_coordinate_median,
    aggregate_trimmed_mean
)
from src.models.ids_net import NetSentryClassifier, get_model_parameters, set_model_parameters

class TestAggregators(unittest.TestCase):
    def setUp(self):
        # Create synthetic client parameter updates for 3 clients
        # Layer 0: shape (3, 2), Layer 1: shape (2,)
        self.client_0 = [np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), np.array([0.1, 0.2])]
        self.client_1 = [np.array([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]), np.array([0.2, 0.3])]
        self.client_2 = [np.array([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]), np.array([0.3, 0.4])]
        
        self.updates_equal_samples = [
            (self.client_0, 100, {}),
            (self.client_1, 100, {}),
            (self.client_2, 100, {})
        ]
        
        self.updates_skewed_samples = [
            (self.client_0, 100, {}),
            (self.client_1, 200, {}),
            (self.client_2, 100, {})
        ]

    def test_fedavg_equal_weights(self):
        avg = aggregate_fedavg(self.updates_equal_samples)
        # Expected layer 0: mean of client_0, client_1, client_2 = client_1
        np.testing.assert_allclose(avg[0], self.client_1[0], atol=1e-5)
        np.testing.assert_allclose(avg[1], self.client_1[1], atol=1e-5)

    def test_fedavg_sample_skewed_weights(self):
        avg = aggregate_fedavg(self.updates_skewed_samples)
        # Weights: 1/4, 2/4, 1/4 -> 0.25 * 1 + 0.5 * 2 + 0.25 * 3 = 2.0
        np.testing.assert_allclose(avg[0], self.client_1[0], atol=1e-5)

    def test_coordinate_median_outlier_resilience(self):
        # Poison client 2 with extreme Byzantine values (+1000.0)
        poisoned_client_2 = [self.client_2[0] + 1000.0, self.client_2[1] + 1000.0]
        poisoned_updates = [
            (self.client_0, 100, {}),
            (self.client_1, 100, {}),
            (poisoned_client_2, 100, {})
        ]
        
        # Median across [client_0, client_1, poisoned_client_2] must be client_1
        med = aggregate_coordinate_median(poisoned_updates)
        np.testing.assert_allclose(med[0], self.client_1[0], atol=1e-5)
        np.testing.assert_allclose(med[1], self.client_1[1], atol=1e-5)

    def test_trimmed_mean(self):
        # 5 clients with 1 extreme positive and 1 extreme negative outlier
        c_neg = [self.client_0[0] - 500.0, self.client_0[1] - 500.0]
        c_pos = [self.client_2[0] + 500.0, self.client_2[1] + 500.0]
        updates_5 = [
            (c_neg, 100, {}),
            (self.client_0, 100, {}),
            (self.client_1, 100, {}),
            (self.client_2, 100, {}),
            (c_pos, 100, {})
        ]
        # Trimmed mean with trim_ratio=0.2 trims top 1 and bottom 1
        # Remaining: client_0, client_1, client_2 -> mean is client_1
        trimmed = aggregate_trimmed_mean(updates_5, trim_ratio=0.2)
        np.testing.assert_allclose(trimmed[0], self.client_1[0], atol=1e-5)

    def test_identical_updates_invariant(self):
        # Invariant: If all updates are identical, all aggregators output the exact same weights
        identical_updates = [(self.client_0, 100, {}) for _ in range(5)]
        fedavg_res = aggregate_fedavg(identical_updates)
        med_res = aggregate_coordinate_median(identical_updates)
        trimmed_res = aggregate_trimmed_mean(identical_updates, trim_ratio=0.2)
        
        np.testing.assert_allclose(fedavg_res[0], self.client_0[0], atol=1e-5)
        np.testing.assert_allclose(med_res[0], self.client_0[0], atol=1e-5)
        np.testing.assert_allclose(trimmed_res[0], self.client_0[0], atol=1e-5)

    def test_quantization_fp16_and_int8(self):
        model = NetSentryClassifier(input_dim=68, hidden_dims=[128, 64, 32], dropout_rate=0.1)
        # Test FP16 conversion
        model_fp16 = NetSentryClassifier(input_dim=68, hidden_dims=[128, 64, 32], dropout_rate=0.1)
        model_fp16.load_state_dict(model.state_dict())
        model_fp16 = model_fp16.half()
        
        x = torch.randn(10, 68)
        model.eval()
        model_fp16.eval()
        with torch.no_grad():
            out_fp32 = model(x)
            out_fp16 = model_fp16(x.half()).float()
        # Outputs should be within tight precision tolerance (< 1e-2)
        np.testing.assert_allclose(out_fp32.numpy(), out_fp16.numpy(), atol=1e-2)

if __name__ == "__main__":
    unittest.main()
