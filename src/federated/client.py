"""
NetSentry-FL: Flower Client Implementation with FedProx Support (Month 5)
Represents a participating organization (Client A, B, or C).
Trains local PyTorch model on strictly private network traffic.
Supports:
1. Standard local SGD (FedAvg)
2. Proximal regularization term (mu / 2) * ||w - w_t||^2 (FedProx)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import flwr as fl
import numpy as np
from typing import Dict, List, Tuple, Any

from src.models.ids_net import NetSentryClassifier, get_model_parameters, set_model_parameters

class NetSentryClient(fl.client.NumPyClient):
    """
    Simulated Federated Organization Node with FedProx Regularization.
    """
    def __init__(self, client_id: int, X_local: np.ndarray, y_local: np.ndarray, input_dim: int, device: str = "cpu"):
        self.client_id = client_id
        self.device = device
        self.input_dim = input_dim
        
        # Local model
        self.model = NetSentryClassifier(input_dim=input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.1).to(device)
        
        # Local data tensors
        self.X_local = torch.from_numpy(X_local).float()
        self.y_local = torch.from_numpy(y_local).float()
        self.num_samples = len(X_local)
        
        # Balanced loss calculation with numerical stability guard
        pos_samples = self.y_local.sum().item()
        neg_samples = self.num_samples - pos_samples
        if pos_samples == 0:
            pos_weight = torch.tensor([1.0]).to(device)
        else:
            raw_wt = neg_samples / pos_samples
            pos_weight = torch.tensor([min(raw_wt, 10.0)]).to(device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        return get_model_parameters(self.model)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[List[np.ndarray], int, Dict]:
        # Synchronize local model with incoming global parameters from server
        set_model_parameters(self.model, parameters)
        
        epochs = int(config.get("local_epochs", 2))
        batch_size = int(config.get("batch_size", 64))
        lr = float(config.get("lr", 0.001))
        proximal_mu = float(config.get("proximal_mu", 0.0))  # FedProx mu parameter
        
        # Freeze a copy of global parameters for the FedProx proximal penalty
        global_params_tensors = []
        if proximal_mu > 0.0:
            for p in self.model.parameters():
                global_params_tensors.append(p.detach().clone())
                
        loader = DataLoader(
            TensorDataset(self.X_local, self.y_local),
            batch_size=batch_size,
            shuffle=True
        )
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
        self.model.train()
        total_loss = 0.0
        for _ in range(epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)
                
                # Add FedProx proximal penalty: (mu / 2) * sum(||w - w_t||^2)
                if proximal_mu > 0.0:
                    proximal_term = 0.0
                    for param, global_param in zip(self.model.parameters(), global_params_tensors):
                        proximal_term += torch.sum((param - global_param) ** 2)
                    loss += (proximal_mu / 2.0) * proximal_term
                    
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        avg_loss = total_loss / (len(loader) * epochs)
        return get_model_parameters(self.model), self.num_samples, {"client_id": self.client_id, "train_loss": avg_loss}

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict]:
        set_model_parameters(self.model, parameters)
        self.model.eval()
        
        with torch.no_grad():
            logits = self.model(self.X_local.to(self.device))
            loss = self.criterion(logits, self.y_local.to(self.device)).item()
            preds = (torch.sigmoid(logits).cpu().numpy() >= 0.5).astype(int)
            accuracy = float((preds == self.y_local.cpu().numpy()).mean())
            
        return float(loss), self.num_samples, {"accuracy": accuracy}
