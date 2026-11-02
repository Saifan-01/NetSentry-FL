"""
NetSentry-FL: PyTorch Neural Network for Intrusion Detection
Multi-Layer Perceptron (MLP) architecture suitable for both
centralized deep learning and federated client-server aggregation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List

class NetSentryClassifier(nn.Module):
    """
    Feed-Forward Neural Network for Network Flow Intrusion Detection.
    Features:
    - Input dimension matches cleaned network features (e.g. 62)
    - Fully connected layers with BatchNorm & Dropout for regularization
    - LeakyReLU activation for robust gradient flow
    - Single logit output for binary classification
    """
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32], dropout_rate: float = 0.2, norm_type: str = "batch_norm"):
        super(NetSentryClassifier, self).__init__()
        
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            if norm_type == "group_norm":
                layers.append(nn.GroupNorm(num_groups=4, num_channels=h_dim))
            elif norm_type == "layer_norm":
                layers.append(nn.LayerNorm(h_dim))
            elif norm_type == "none":
                pass
            else:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.LeakyReLU(negative_slope=0.1))
            layers.append(nn.Dropout(p=dropout_rate))
            curr_dim = h_dim
            
        layers.append(nn.Linear(curr_dim, 1))
        self.network = nn.Sequential(*layers)
        self.norm_type = norm_type
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and x.size(0) == 1:
            for m in self.network.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()
            out = self.network(x).squeeze(-1)
            for m in self.network.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.train()
            return out
        return self.network(x).squeeze(-1)

def get_model_parameters(model: nn.Module) -> List[np.ndarray]:
    """Extracts model weights as a list of NumPy arrays for Flower serialization."""
    import numpy as np
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_model_parameters(model: nn.Module, parameters: List) -> None:
    """Sets PyTorch model parameters from a list of NumPy arrays (from Flower server)."""
    import torch
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
