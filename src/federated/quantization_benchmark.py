"""
NetSentry-FL: Quantization & Communication Efficiency Benchmark
Evaluates model compression on the Global Federated Model:
- FP32 (Full precision baseline: 4 bytes/parameter)
- FP16 (Half precision: 2 bytes/parameter, 50% bandwidth reduction)
- INT8 (Dynamic integer quantization: 1 byte/parameter, 75% bandwidth reduction)

Evaluated across the 7,292 held-out test flows across all 5 seeds.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os
import io
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, precision_recall_curve, auc

from src.models.ids_net import NetSentryClassifier, get_model_parameters, set_model_parameters
from src.federated.client import NetSentryClient
from src.federated.partitioners import create_extreme_attack_silos
from src.models.evaluate_month5_tightened import run_fl_simulation, evaluate_metrics

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def get_model_size_kb(model: nn.Module) -> float:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell() / 1024.0

def evaluate_quantized_model(model: nn.Module, X_test: np.ndarray, y_test: np.ndarray, is_fp16: bool = False):
    model.eval()
    with torch.no_grad():
        x_tensor = torch.from_numpy(X_test)
        if is_fp16:
            x_tensor = x_tensor.half()
        logits = model(x_tensor).float()
        probs = torch.sigmoid(logits).numpy()
    return evaluate_metrics(y_test, probs)

def main():
    torch.set_num_threads(min(12, os.cpu_count() or 4))
    print("="*85, flush=True)
    print("      NETSENTRY-FL: MODEL QUANTIZATION & BANDWIDTH EFFICIENCY BENCHMARK      ", flush=True)
    print("="*85, flush=True)

    X_train = np.load(PROCESSED_DIR / "X_train.npy").astype(np.float32)
    y_train = np.load(PROCESSED_DIR / "y_train.npy").astype(np.float32)
    X_val = np.load(PROCESSED_DIR / "X_val.npy").astype(np.float32)
    y_val = np.load(PROCESSED_DIR / "y_val.npy").astype(np.float32)
    X_test = np.load(PROCESSED_DIR / "X_test.npy").astype(np.float32)
    y_test = np.load(PROCESSED_DIR / "y_test.npy").astype(np.float32)
    train_meta = pd.read_csv(PROCESSED_DIR / "train_metadata.csv")
    input_dim = X_train.shape[1]

    seeds = [42, 101, 777, 1234, 5678]
    
    fp32_runs = []
    fp16_runs = []
    int8_runs = []
    
    fp32_size = None
    fp16_size = None
    int8_size = None

    for seed in seeds:
        print(f"\n--- Running Seed {seed} ---", flush=True)
        # Train a federated model (IID benchmark)
        idx_perm = np.random.RandomState(seed).permutation(len(X_train))
        splits = np.array_split(idx_perm, 3)
        clients_dict = {cid: {"X": X_train[splits[cid]], "y": y_train[splits[cid]]} for cid in range(3)}
        
        fed_model, _, _, _ = run_fl_simulation(
            clients_dict, input_dim, X_val, y_val, X_test, y_test,
            num_rounds=15, local_epochs=2, lr=0.001, seed=seed, setup_name="Quant_Train"
        )
        
        # 1. FP32 Baseline
        m_fp32 = evaluate_quantized_model(fed_model, X_test, y_test)
        fp32_runs.append(m_fp32)
        if fp32_size is None:
            fp32_size = get_model_size_kb(fed_model)
            
        # 2. FP16 Quantization
        fed_model_fp16 = NetSentryClassifier(input_dim=input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.1)
        fed_model_fp16.load_state_dict(fed_model.state_dict())
        fed_model_fp16 = fed_model_fp16.half()
        m_fp16 = evaluate_quantized_model(fed_model_fp16, X_test, y_test, is_fp16=True)
        fp16_runs.append(m_fp16)
        if fp16_size is None:
            fp16_size = get_model_size_kb(fed_model_fp16)
            
        # 3. INT8 Dynamic Quantization
        fed_model_int8 = torch.ao.quantization.quantize_dynamic(
            fed_model, {nn.Linear}, dtype=torch.qint8
        )
        m_int8 = evaluate_quantized_model(fed_model_int8, X_test, y_test)
        int8_runs.append(m_int8)
        if int8_size is None:
            int8_size = get_model_size_kb(fed_model_int8)
            
        print(f"  FP32: F1={m_fp32['f1']:.4f} | FP16: F1={m_fp16['f1']:.4f} | INT8: F1={m_int8['f1']:.4f}", flush=True)

    def summarize_format(runs, name, size):
        savings = f"{(1.0 - size / fp32_size) * 100:.1f}%" if size != fp32_size else "0.0%"
        f1_vals = [r["f1"] for r in runs]
        rec_vals = [r["recall"] for r in runs]
        prec_vals = [r["precision"] for r in runs]
        fpr_vals = [r["fpr"] for r in runs]
        return {
            "Precision_Format": name,
            "Model_Size_KB": round(size, 2),
            "Bandwidth_Savings": savings,
            "F1_Mean": round(float(np.mean(f1_vals)), 4),
            "F1_Std": round(float(np.std(f1_vals)), 4),
            "Recall_Mean": round(float(np.mean(rec_vals)), 4),
            "Recall_Std": round(float(np.std(rec_vals)), 4),
            "Precision_Mean": round(float(np.mean(prec_vals)), 4),
            "Precision_Std": round(float(np.std(prec_vals)), 4),
            "FPR_Mean": round(float(np.mean(fpr_vals)), 4),
            "FPR_Std": round(float(np.std(fpr_vals)), 4),
            "F1": f"{np.mean(f1_vals):.4f} +/- {np.std(f1_vals):.4f}",
            "Recall": f"{np.mean(rec_vals):.4f} +/- {np.std(rec_vals):.4f}",
            "Precision": f"{np.mean(prec_vals):.4f} +/- {np.std(prec_vals):.4f}",
            "False positive rate": f"{np.mean(fpr_vals):.4f} +/- {np.std(fpr_vals):.4f}"
        }

    records = [
        summarize_format(fp32_runs, "FP32 (Full Precision Baseline)", fp32_size),
        summarize_format(fp16_runs, "FP16 (Half Precision)", fp16_size),
        summarize_format(int8_runs, "INT8 (Dynamic Integer Quantization)", int8_size)
    ]
    df = pd.DataFrame(records)
    df.to_csv(EXPERIMENTS_DIR / "quantization_benchmark.csv", index=False)
    
    print("\n" + "="*95, flush=True)
    print("                MODEL QUANTIZATION & COMPRESSION BENCHMARK (5 SEEDS)                ", flush=True)
    print("="*95, flush=True)
    print(df[["Precision_Format", "Model_Size_KB", "Bandwidth_Savings", "F1", "Recall", "False positive rate"]].to_string(index=False), flush=True)
    print("="*95 + "\n", flush=True)

if __name__ == "__main__":
    main()
