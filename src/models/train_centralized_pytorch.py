"""
NetSentry-FL: Centralized PyTorch Neural Network Training
Trains the NetSentryClassifier on the pooled training dataset.
Serves as the empirical upper-bound benchmark for comparison against Federated Learning.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.models.ids_net import NetSentryClassifier

PROCESSED_DIR = Path("data/processed")
EXPERIMENTS_DIR = Path("experiments")

def train_centralized(
    epochs: int = 15,
    batch_size: int = 128,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    print(f"[*] Training Centralized PyTorch Model on device: {device}")
    
    # Load data
    X_train = np.load(PROCESSED_DIR / "X_train.npy").astype(np.float32)
    y_train = np.load(PROCESSED_DIR / "y_train.npy").astype(np.float32)
    X_val = np.load(PROCESSED_DIR / "X_val.npy").astype(np.float32)
    y_val = np.load(PROCESSED_DIR / "y_val.npy").astype(np.float32)
    X_test = np.load(PROCESSED_DIR / "X_test.npy").astype(np.float32)
    y_test = np.load(PROCESSED_DIR / "y_test.npy").astype(np.float32)

    # Class imbalance weight for BCEWithLogitsLoss
    pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / (y_train.sum() + 1e-5)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True
    )
    
    model = NetSentryClassifier(input_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        if epoch % 5 == 0 or epoch == epochs:
            print(f"    Epoch {epoch:2d}/{epochs:2d} - Loss: {avg_loss:.4f}")

    # Evaluate on held-out test set
    model.eval()
    with torch.no_grad():
        test_logits = model(torch.from_numpy(X_test).to(device))
        probs = torch.sigmoid(test_logits).cpu().numpy()
        preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    roc = roc_auc_score(y_test, probs)

    print("\n========================================================")
    print("  Centralized PyTorch Neural Network Test Results")
    print("========================================================")
    print(f"  Accuracy:        {acc * 100:.2f}%")
    print(f"  Precision:       {prec * 100:.2f}%")
    print(f"  Recall:          {rec * 100:.2f}%")
    print(f"  F1-Score:        {f1:.4f}")
    print(f"  ROC-AUC:         {roc:.4f}")
    print("========================================================\n")

    # Save model weights
    torch.save(model.state_dict(), EXPERIMENTS_DIR / "centralized_ids_pytorch.pth")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc}

if __name__ == "__main__":
    train_centralized()
