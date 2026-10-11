"""
NetSentry-FL: Flower Client Implementation
Represents a participating organization (Client A, B, or C).
Trains the local PyTorch model on strictly private network traffic.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import flwr as fl
import numpy as np
from typing import Dict, List, Tuple
from src.models.ids_net import NetSentryClassifier, get_model_parameters, set_model_parameters

class NetSentryClient(fl.client.NumPyClient):
    """
    Simulated Federated Organization Node.
    Trains on local data partition; transmits only weight updates to server.
    """
    def __init__(self, client_id: int, X_local: np.ndarray, y_local: np.ndarray, input_dim: int, device: str = "cpu"):
        self.client_id = client_id
        self.device = device
        self.input_dim = input_dim
        
        # Local model
        self.model = NetSentryClassifier(input_dim=input_dim).to(device)
        
        # Local data
        self.X_local = torch.from_numpy(X_local).float()
        self.y_local = torch.from_numpy(y_local).float()
        self.num_samples = len(X_local)
        
        # Handle class imbalance locally
        pos_samples = self.y_local.sum().item()
        neg_samples = self.num_samples - pos_samples
        pos_weight = torch.tensor([neg_samples / (pos_samples + 1e-5)]).to(device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        return get_model_parameters(self.model)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict]:
        # Synchronize local model with incoming global parameters from server
        set_model_parameters(self.model, parameters)
        
        epochs = int(config.get("local_epochs", 2))
        batch_size = int(config.get("batch_size", 64))
        lr = float(config.get("lr", 0.001))
        
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
