"""
NetSentry-FL: Interactive Streamlit Demonstration Dashboard
Features:
1. Live Network Flow Inference: Upload a CSV of network flows and get Benign/Attack predictions using the Global Federated Model.
2. Benchmark Results Section: Displays the definitive 3-seed comparison table with metrics.
3. Interactive Visualizations: High-resolution charts for Convergence, Per-Attack Recall, Setup Comparison, and Communication Overhead.
4. System Architecture: Interactive topology diagram showing privacy-preserving Flower FedAvg pipeline.
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.ids_net import NetSentryClassifier

st.set_page_config(
    page_title="NetSentry-FL | Federated Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

EXPERIMENTS_DIR = Path("experiments")
PROCESSED_DIR = Path("data/processed")
CHARTS_DIR = Path("reports/charts")

@st.cache_resource
def load_model_and_scaler():
    feature_names = joblib.load(PROCESSED_DIR / "feature_names.pkl")
    scaler = joblib.load(PROCESSED_DIR / "scaler.pkl")
    
    model = NetSentryClassifier(input_dim=len(feature_names))
    model_path = EXPERIMENTS_DIR / "global_federated_model.pth"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    return model, scaler, feature_names

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=64)
st.sidebar.title("NetSentry-FL")
st.sidebar.markdown("**Privacy-Preserving Federated IDS**")
nav = st.sidebar.radio(
    "Navigation",
    ["1. Live Flow Detection Demo", "2. Benchmark Comparison Table", "3. Experimental Charts", "4. System Architecture & Privacy"]
)

st.sidebar.divider()
st.sidebar.info(
    "**Core Principle:** Raw traffic data never leaves Client A, B, or C. "
    "Only encrypted weight updates are communicated to the FedAvg server."
)

model, scaler, feature_names = load_model_and_scaler()

# PAGE 1: Live Demo
if nav == "1. Live Flow Detection Demo":
    st.title("🛡️ NetSentry-FL: Live Network Intrusion Detection")
    st.markdown("Upload raw or pre-processed network flow records to classify traffic as **Benign** or **Malicious (Attack)** using the aggregated **Global Federated Model**.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader("Upload Network Traffic CSV (e.g. CIC-IDS2017 format)", type=["csv"])
    with col2:
        st.markdown("#### Sample Test Flow Generator")
        if st.button("Load 50 Sample Held-out Test Flows"):
            raw_sample = pd.read_csv("data/raw/CIC_IDS2017_benchmark_sample.csv").iloc[-50:].copy()
            st.session_state["demo_df"] = raw_sample

    if "demo_df" in st.session_state and uploaded_file is None:
        df = st.session_state["demo_df"]
        st.success("Loaded 50 sample held-out test flows from CIC-IDS2017 benchmark.")
    elif uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"Uploaded CSV with {len(df)} rows.")
    else:
        df = None

    if df is not None:
        st.subheader("Preview Flow Data")
        st.dataframe(df.head(5), use_container_width=True)
        
        # Verify and align features
        available_features = [f for f in feature_names if f in df.columns]
        missing_features = [f for f in feature_names if f not in df.columns]
        
        if len(missing_features) > 0:
            st.warning(f"Note: {len(missing_features)} features missing from CSV, imputing with 0.0.")
            for mf in missing_features:
                df[mf] = 0.0
                
        X_raw = df[feature_names].values.astype(np.float32)
        X_scaled = scaler.transform(X_raw).astype(np.float32)
        
        with torch.no_grad():
            logits = model(torch.from_numpy(X_scaled))
            probs = torch.sigmoid(logits).numpy()
            preds = (probs >= 0.5).astype(int)
            
        df["Threat_Probability"] = np.round(probs * 100, 2)
        df["Predicted_Status"] = ["🚨 ATTACK" if p == 1 else "✅ BENIGN" for p in preds]
        
        st.subheader("Inference Results")
        kpi1, kpi2, kpi3 = st.columns(3)
        total_flows = len(df)
        attack_count = int(preds.sum())
        benign_count = total_flows - attack_count
        
        kpi1.metric("Total Flows Analyzed", total_flows)
        kpi2.metric("Detected Threats (Attacks)", attack_count, delta=f"{attack_count/total_flows*100:.1f}%", delta_color="inverse")
        kpi3.metric("Benign Normal Traffic", benign_count)
        
        st.dataframe(
            df[["Predicted_Status", "Threat_Probability"] + [c for c in ["Destination Port", "Flow Duration", "Total Fwd Packets", "Label"] if c in df.columns]].head(25),
            use_container_width=True
        )

# PAGE 2: Comparison Table
elif nav == "2. Benchmark Comparison Table":
    st.title("📊 The Main Result: 3-Seed Rigorous Benchmark Comparison")
    st.markdown("All configurations are evaluated on the exact same **8,000 held-out test flows** and averaged over **3 distinct seeds (42, 101, 777)**.")
    
    comp_path = EXPERIMENTS_DIR / "rigorous_comparison_table.csv"
    if comp_path.exists():
        comp_df = pd.read_csv(comp_path)
        st.dataframe(
            comp_df.style.highlight_max(subset=["Precision", "Recall", "F1", "PR-AUC"], color="#c6eecd")
                         .highlight_min(subset=["False positive rate"], color="#d4edda"),
            use_container_width=True
        )
    
    st.markdown("### Key Empirical Findings:")
    st.markdown("""
    1. **Federated Approaches Centralized:** Federated Learning achieves high precision and recall comparable to Centralized training without exposing any raw private logs to the server.
    2. **Local Blindspots in Data Silos:** When clients train in isolation (`Local-only Client A / B / C`), each client has strong detection for its local attack types but completely fails on attack families it never observed.
    3. **Non-IID Convergence Penalty:** Heterogeneous traffic distributions (Non-IID) cause client drift and require more communication rounds to converge compared to uniform (IID) distributions.
    """)

# PAGE 3: Experimental Charts
elif nav == "3. Experimental Charts":
    st.title("📈 Evaluation Visualizations")
    
    tab1, tab2, tab3, tab4 = st.tabs(["1. Federated Convergence", "2. Per-Attack-Type Recall", "3. Metric Comparison", "4. Communication Cost"])
    
    with tab1:
        st.subheader("Federated Convergence: IID vs. Non-IID")
        st.image(str(CHARTS_DIR / "convergence_iid_vs_non_iid.png"), use_container_width=True)
        st.caption("IID aggregation reaches fast convergence, whereas Non-IID experiences gradient divergence in early rounds before stabilizing.")
        
    with tab2:
        st.subheader("Per-Attack Recall: Local Blindspots vs. Federated Generalization")
        st.image(str(CHARTS_DIR / "per_attack_recall.png"), use_container_width=True)
        st.caption("Notice how Local Client A misses DoS & DDoS attacks, while Federated models aggregate client intelligence to recognize all attack classes.")
        
    with tab3:
        st.subheader("Comparative Metric Performance Across Setups")
        st.image(str(CHARTS_DIR / "setup_comparison_metrics.png"), use_container_width=True)
        
    with tab4:
        st.subheader("Communication Overhead (Total Bytes Exchanged)")
        st.image(str(CHARTS_DIR / "communication_cost.png"), use_container_width=True)
        comm_df = pd.read_csv(EXPERIMENTS_DIR / "communication_overhead.csv")
        st.table(comm_df)

# PAGE 4: System Architecture
elif nav == "4. System Architecture & Privacy":
    st.title("🏗️ System Architecture & Threat Model")
    st.markdown("""
    ```
                    ┌──────────────────────────────────────────────┐
                    │          Flower FedAvg Server                │
                    │        Global IDS Model (MLP)                │
                    └───────┬──────────────┬──────────────┬────────┘
                            │              │              │
             Global Weights │              │              │ Global Weights
                   (73.7 KB)│              │              │ (73.7 KB)
                            ▼              ▼              ▼
                     ┌──────────────┐┌──────────────┐┌──────────────┐
                     │   Client A   ││   Client B   ││   Client C   │
                     │  (Bank Silo) ││(Health Silo) ││ (Cloud Silo) │
                     └──────┬───────┘└──────┬───────┘└──────┬───────┘
                            │               │               │
                     Local SGD (2 ep)Local SGD (2 ep)Local SGD (2 ep)
                     [PRIVATE DATA]  [PRIVATE DATA]  [PRIVATE DATA]
                     (Never leaves)  (Never leaves)  (Never leaves)
    ```
    """)
    st.markdown("""
    ### Privacy & Security Guarantees:
    - **Raw Data Locality:** Raw network flows, IP addresses, and timestamps remain strictly inside the organizational boundary.
    - **Flower Orchestration:** Clients download global parameter vectors $\theta_t$, perform local training updates $\Delta \theta_k$, and upload updated vectors.
    - **Residual Privacy Risks:** Model inversion and membership inference attacks remain theoretical risks if an adversary inspects raw weight differentials.
    """)
