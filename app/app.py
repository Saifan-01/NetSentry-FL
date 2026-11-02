"""
NetSentry-FL: Interactive Streamlit Demonstration Dashboard
Professional Academic UI
Strictly evaluated on real CIC-IDS2017 network traffic and the trained Global Federated Model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.ids_net import NetSentryClassifier

st.set_page_config(
    page_title="NetSentry-FL | Federated Intrusion Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEMO_DIR = PROJECT_ROOT / "data" / "demo_samples"
CHARTS_DIR = PROJECT_ROOT / "reports" / "charts"

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

model, scaler, feature_names = load_model_and_scaler()

# Sidebar
st.sidebar.title("NetSentry-FL")
st.sidebar.caption("Privacy-Preserving Federated IDS")
nav = st.sidebar.radio(
    "Navigation Menu",
    [
        "Live Traffic Classifier",
        "5-Seed Benchmark Comparison",
        "Month 6: Security, Privacy & Quantization",
        "Traditional Centralized Baselines",
        "Convergence and Cost Charts",
        "System Architecture"
    ]
)

st.sidebar.divider()
st.sidebar.markdown("### Model Specifications")
st.sidebar.markdown(f"""
- **Architecture:** 4-Layer PyTorch MLP
- **Input Features:** {len(feature_names)} Real CIC-IDS2017 metrics
- **Framework:** Flower FedAvg
- **Privacy Guarantee:** Raw flow logs remain on client premises
""")

# PAGE 1: LIVE TRAFFIC CLASSIFIER
if nav == "Live Traffic Classifier":
    st.title("NetSentry-FL: Live Network Traffic Intrusion Detection")
    st.markdown(
        "Upload real CIC-IDS2017 bidirectional network flow records (or select a pre-loaded enterprise traffic stream) "
        "to classify traffic into **Benign** or **Malicious (Attack)** using the aggregated **Global Federated Model**."
    )
    
    st.divider()
    
    col_input1, col_input2 = st.columns([1.2, 1])
    
    with col_input1:
        st.subheader("Option A: Upload Traffic CSV")
        uploaded_file = st.file_uploader(
            "Upload CSV containing network flow features",
            type=["csv"],
            help="Accepts CIC-IDS2017 CSV files with standard network flow metrics (Destination Port, Flow Duration, etc.)"
        )
        
    with col_input2:
        st.subheader("Option B: Select Pre-loaded Scenario")
        scenario_choice = st.selectbox(
            "Select real CIC-IDS2017 traffic profile:",
            [
                "None (Use Uploaded File)",
                "Mixed Enterprise Traffic (100 flows: Normal + DoS + PortScan + Web)",
                "Active Cyberattack Influx (50 flows: Multi-vector attack)",
                "Clean Normal Office Hours Traffic (50 flows: 100% Benign)"
            ]
        )
        
        scenario_map = {
            "Mixed Enterprise Traffic (100 flows: Normal + DoS + PortScan + Web)": "demo_mixed_enterprise_traffic.csv",
            "Active Cyberattack Influx (50 flows: Multi-vector attack)": "demo_attack_incident_traffic.csv",
            "Clean Normal Office Hours Traffic (50 flows: 100% Benign)": "demo_clean_business_traffic.csv"
        }

    df = None
    source_label = ""
    
    if scenario_choice in scenario_map and uploaded_file is None:
        selected_file = DEMO_DIR / scenario_map[scenario_choice]
        if selected_file.exists():
            df = pd.read_csv(selected_file)
            source_label = scenario_choice
            st.success(f"Loaded {len(df)} real CIC-IDS2017 flows from scenario: {scenario_choice}")
    elif uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        source_label = f"Uploaded File: {uploaded_file.name}"
        st.success(f"Successfully loaded {len(df)} flows from {uploaded_file.name}")

    if df is not None:
        st.divider()
        st.markdown(f"### Real-Time Inference: {source_label}")
        
        # Strict Feature Alignment and Validation Check
        missing_features = [f for f in feature_names if f not in df.columns]
        if len(missing_features) > 5:
            st.error(
                f"Feature Alignment Error: The uploaded dataset is missing {len(missing_features)} of the {len(feature_names)} "
                f"required CIC-IDS2017 flow features. Missing columns include: {missing_features[:10]}... "
                "Please upload a valid CIC-IDS2017 flow record CSV."
            )
            st.stop()
        elif len(missing_features) > 0:
            st.warning(f"Note: {len(missing_features)} minor features were missing and imputed with 0.0: {missing_features}")
            for mf in missing_features:
                df[mf] = 0.0
                
        threshold = st.slider(
            "Threat Decision Threshold (tau)",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05,
            help="Flows with Threat Probability >= tau are flagged as Attacks. Defaults to 0.50."
        )
        
        # Enforce exact column order as scaler training
        X_raw = df[feature_names].values.astype(np.float32)
        X_scaled = scaler.transform(X_raw).astype(np.float32)
        
        with torch.no_grad():
            logits = model(torch.from_numpy(X_scaled))
            probs = torch.sigmoid(logits).numpy()
            preds = (probs >= threshold).astype(int)
            
        results_df = df.copy()
        results_df["Threat_Probability"] = np.round(probs * 100, 2)
        results_df["Predicted_Status"] = ["ATTACK" if p == 1 else "BENIGN" for p in preds]
        
        total_count = len(results_df)
        attack_count = int(preds.sum())
        benign_count = total_count - attack_count
        attack_pct = (attack_count / total_count) * 100
        
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Analyzed Flows", f"{total_count:,}")
        kpi2.metric(
            "Detected Attacks",
            f"{attack_count:,}",
            delta=f"{attack_pct:.1f}% attack rate",
            delta_color="inverse" if attack_count > 0 else "normal"
        )
        kpi3.metric("Benign Normal Flows", f"{benign_count:,}")
        kpi4.metric("Avg Threat Probability", f"{probs.mean() * 100:.1f}%")
        
        if attack_count > 0:
            st.error(f"Security Alert: Global Federated IDS flagged {attack_count} network flows matching intrusion patterns.")
        else:
            st.success("Network Clear: All analyzed flow records exhibit normal benign operational behavior.")

        st.subheader("Flow-by-Flow Classification Results")
        filter_status = st.radio("Filter Flows:", ["All Flows", "Attacks Only", "Benign Only"], horizontal=True)
        
        view_df = results_df
        if filter_status == "Attacks Only":
            view_df = results_df[results_df["Predicted_Status"] == "ATTACK"]
        elif filter_status == "Benign Only":
            view_df = results_df[results_df["Predicted_Status"] == "BENIGN"]

        display_cols = ["Predicted_Status", "Threat_Probability"]
        for c in ["Category", "Destination Port", "Flow Duration", "Total Fwd Packets", "Flow Bytes/s", "Label"]:
            if c in view_df.columns and c not in display_cols:
                display_cols.append(c)
                
        st.dataframe(
            view_df[display_cols],
            column_config={
                "Predicted_Status": st.column_config.TextColumn("Status", width="medium"),
                "Threat_Probability": st.column_config.ProgressColumn(
                    "Threat Probability (%)",
                    format="%.2f%%",
                    min_value=0,
                    max_value=100
                ),
                "Destination Port": st.column_config.NumberColumn("Port", format="%d"),
                "Flow Duration": st.column_config.NumberColumn("Duration (us)", format="%d"),
                "Total Fwd Packets": st.column_config.NumberColumn("Fwd Packets", format="%d"),
                "Flow Bytes/s": st.column_config.NumberColumn("Bytes/s", format="%.2f"),
                "Category": st.column_config.TextColumn("Ground Truth Category")
            },
            use_container_width=True,
            hide_index=True
        )
        
        csv_bytes = view_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Classified Flow Log (CSV)",
            data=csv_bytes,
            file_name="netsentry_fl_classified_traffic.csv",
            mime="text/csv"
        )

# PAGE 2: 5-SEED BENCHMARK COMPARISON
elif nav in ["5-Seed Benchmark Comparison", "3-Seed Benchmark Comparison"]:
    st.title("The Definitive 5-Seed Comparison Table (Mean +/- Spread)")
    st.markdown(
        "All configurations are evaluated on the exact same **7,292 real held-out test flows** (20% split) "
        "and averaged over **5 distinct random seeds (42, 101, 777, 1234, 5678)** with strict final-round evaluation (R=15)."
    )
    
    comp_df = pd.read_csv(EXPERIMENTS_DIR / "rigorous_comparison_table.csv")
    display_cols = ["Setup", "Precision", "Recall", "F1", "PR-AUC", "False positive rate"]
    st.dataframe(
        comp_df[display_cols],
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("### Critical Research Insights:")
    st.markdown("""
    - **Local Blindspots in Isolated Silos:** Clients trained only on localized threats fail completely when exposed to new attacks. Client B (PortScan silo) achieves **0.00% recall on DoS and Web Attacks**. Client A (DoS silo) achieves **1.16% recall on PortScan**.
    - **Federated Generalization:** Federated IID **matches centralized performance** (0.9894 vs 0.9883 F1) and eliminates local blindspots across all attack categories.
    - **Extreme Silo Dynamics:** On extreme disjoint silos, standard FedAvg achieves F1 0.8182 +/- 0.0856. FedProx (mu*=1.0) improves the mean F1 to **0.8750 +/- 0.0559** and recall to **78.99% +/- 8.75%**, winning in **4 out of 5 seeds** (paired differences: -0.0302, +0.2096, +0.0330, +0.0637, +0.0080).
    - **Dirichlet Skew Finding:** On continuous Dirichlet label skew (alpha=0.1), performance differences between FedAvg (0.9519 +/- 0.0336) and FedProx (0.9417 +/- 0.0529) fall within mutual standard error bounds, indicating no statistically clear difference on continuous skew.
    """)
    
    st.divider()
    st.subheader("Per-Attack-Type Recall Breakdown (Mean +/- Spread, 5 Seeds)")
    if (EXPERIMENTS_DIR / "per_attack_recall.csv").exists():
        att_df = pd.read_csv(EXPERIMENTS_DIR / "per_attack_recall.csv")
        st.dataframe(
            att_df[["Attack", "Model", "Recall_Display", "Total_Flows"]].rename(columns={"Recall_Display": "Detection Recall (Mean +/- Std)"}),
            use_container_width=True,
            hide_index=True
        )

# PAGE 3: MONTH 6 ADVANCED BENCHMARKS
elif nav == "Month 6: Security, Privacy & Quantization":
    st.title("Month 6: Advanced Security, Privacy & Compression")
    st.markdown(
        "Evaluations spanning bandwidth reduction via quantization, Byzantine robustness against label poisoning, "
        "and formal mathematical Differential Privacy (DP-SGD via Opacus) across 5 independent seeds."
    )
    
    m6_tab1, m6_tab2, m6_tab3, m6_tab4 = st.tabs([
        "1. Model Quantization",
        "2. Byzantine Defense",
        "3. Differential Privacy (DP-SGD)",
        "4. Minority Web Attack Ablation"
    ])
    
    with m6_tab1:
        st.subheader("Model Quantization & Bandwidth Savings (5 Seeds)")
        st.markdown(
            "Compares full-precision FP32 parameters against half-precision FP16 and dynamic 8-bit integer quantization (INT8):"
        )
        if (EXPERIMENTS_DIR / "quantization_benchmark.csv").exists():
            q_df = pd.read_csv(EXPERIMENTS_DIR / "quantization_benchmark.csv")
            st.dataframe(
                q_df[["Precision_Format", "Model_Size_KB", "Bandwidth_Savings", "F1", "Recall", "False positive rate"]],
                use_container_width=True,
                hide_index=True
            )
        if (CHARTS_DIR / "quantization_compression.png").exists():
            st.image(str(CHARTS_DIR / "quantization_compression.png"), use_container_width=True)
        st.caption("Result: INT8 dynamic quantization slashes bandwidth payload by 61.6% with 0.00 degradation in test F1.")
        
    with m6_tab2:
        st.subheader("Byzantine Robustness against 1 Label-Flipping Malicious Client")
        st.markdown(
            "Threat Model: 1 compromised client actively inverts binary labels (0 <-> 1). Evaluated on 3 clients (33.3% malicious) and 5 clients (20.0% malicious):"
        )
        if (EXPERIMENTS_DIR / "robust_aggregation_benchmark.csv").exists():
            r_df = pd.read_csv(EXPERIMENTS_DIR / "robust_aggregation_benchmark.csv")
            st.dataframe(
                r_df[["Experiment", "Malicious_Ratio", "Aggregator", "F1", "Recall", "False_Alarm_Rate"]],
                use_container_width=True,
                hide_index=True
            )
        if (CHARTS_DIR / "poisoning_robustness.png").exists():
            st.image(str(CHARTS_DIR / "poisoning_robustness.png"), use_container_width=True)
        st.caption("Coordinate Median and Trimmed Mean suppress false alarms back to ~1.2% and completely restore F1-score.")
        
    with m6_tab3:
        st.subheader("Differential Privacy: Privacy-Utility Tradeoff (Opacus DP-SGD, delta=1e-5)")
        st.markdown(
            "Per-client DP-SGD with clipping norm C=1.0 and noise multiplier sweep sigma in {0.5, 1.0, 2.0} using GroupNorm MLP:"
        )
        if (EXPERIMENTS_DIR / "dp_privacy_utility_tradeoff.csv").exists():
            dp_df = pd.read_csv(EXPERIMENTS_DIR / "dp_privacy_utility_tradeoff.csv")
            st.dataframe(
                dp_df[["Configuration", "Epsilon_Display", "F1", "Recall", "False_Alarm_Rate", "DoS_Recall_Mean", "Web_Attack_Recall_Mean"]],
                use_container_width=True,
                hide_index=True
            )
        if (CHARTS_DIR / "dp_epsilon_vs_f1.png").exists():
            st.image(str(CHARTS_DIR / "dp_epsilon_vs_f1.png"), use_container_width=True)
        st.info("Privacy Guarantee Scope: Bounds the probability of an untrusted central server or snooping network observer reconstructing private training flows.")
        
    with m6_tab4:
        st.subheader("Web Attack Minority Class Ablation & Per-Round Dynamics")
        st.markdown(
            "Client C holds only 1,500 attack flows out of 8,969 total attack flows (16.7% of attacks in training). "
            "Under standard sample-weighted FedAvg, updates are diluted. Equal weighting (1/K) and FedProx stabilize learning:"
        )
        if (EXPERIMENTS_DIR / "web_attack_ablation_summary.csv").exists():
            w_df = pd.read_csv(EXPERIMENTS_DIR / "web_attack_ablation_summary.csv")
            st.dataframe(
                w_df[["Strategy", "Test_F1_Display", "Web_Recall_Display"]].rename(columns={
                    "Test_F1_Display": "Test F1-Score (Mean +/- Std)",
                    "Web_Recall_Display": "Web Attack Recall (Mean +/- Std)"
                }),
                use_container_width=True,
                hide_index=True
            )
        if (CHARTS_DIR / "web_attack_recall_per_round.png").exists():
            st.image(str(CHARTS_DIR / "web_attack_recall_per_round.png"), use_container_width=True)
        st.caption("Per-round tracking reveals FedAvg suffers from early learning followed by majority update overwriting; FedProx achieves 84.71% +/- 0.19% recall.")

# PAGE 4: TRADITIONAL CENTRALIZED BASELINES
elif nav == "Traditional Centralized Baselines":
    st.title("Traditional Centralized Baseline Models")
    st.markdown("Baseline evaluation of classical machine learning and deep learning models trained on pooled data:")
    
    base_df = pd.read_csv(EXPERIMENTS_DIR / "centralized_baseline_results.csv")
    st.dataframe(base_df, use_container_width=True, hide_index=True)
    
    st.markdown("""
    - **Random Forest (Centralized Upper Bound):** Achieves near-perfect performance on pooled data (99.78% accuracy, 0.9969 F1, 0.13% false alarm rate), serving as the theoretical upper bound.
    - **Centralized PyTorch MLP (Federated-Compatible Architecture):** Matches Random Forest with 99.09% accuracy, 97.63% precision, 99.84% recall, and 0.9873 F1 using Gaussian Quantile feature transformation.
    - **Logistic Regression:** Reaches 97.46% accuracy and 0.9644 F1, but incurs higher false alarms (2.73% FPR).
    """)
    
    st.divider()
    st.subheader("Architectural Privacy Analysis: Feature Scaling in Federated IDS")
    st.markdown("""
    In standard centralized benchmarks, feature scalers are fit on pooled training data. In real-world cross-silo FL, 
    pooling raw features compromises privacy. NetSentry-FL evaluates three feature scaling paradigms:
    """)
    if (EXPERIMENTS_DIR / "scaling_privacy_comparison.csv").exists():
        scal_df = pd.read_csv(EXPERIMENTS_DIR / "scaling_privacy_comparison.csv")
        st.dataframe(scal_df, use_container_width=True, hide_index=True)
        st.caption("Empirical test on 7,292 real held-out test flows comparing centralized, decentralized, and parameterless transformations.")

# PAGE 4: CONVERGENCE & CHARTS
elif nav == "Convergence and Cost Charts":
    st.title("Experimental Performance and Convergence Charts")
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "1. Federated Convergence",
        "2. Per-Attack Recall",
        "3. Metrics Comparison",
        "4. Client Partitions",
        "5. Quantization & Bandwidth",
        "6. Poisoning Robustness",
        "7. DP Tradeoff Curve"
    ])
    
    with tab1:
        st.subheader("Federated Convergence: Global Model F1-Score per Round")
        st.image(str(CHARTS_DIR / "convergence_iid_vs_non_iid.png"), use_container_width=True)
        st.caption("IID FedAvg converges smoothly, while Non-IID exhibits initial gradient oscillation.")
        
    with tab2:
        st.subheader("Per-Attack Recall Breakdown")
        st.image(str(CHARTS_DIR / "per_attack_recall.png"), use_container_width=True)
        st.caption("Shows how local clients miss attack categories they never experienced, whereas the federated model learns broad coverage.")
        
    with tab3:
        st.subheader("Precision, Recall, and F1 across Setups")
        st.image(str(CHARTS_DIR / "setup_comparison_metrics.png"), use_container_width=True)
        
    with tab4:
        st.subheader("Client Class Distribution Partitions")
        if (CHARTS_DIR / "dirichlet_and_silo_distributions.png").exists():
            st.image(str(CHARTS_DIR / "dirichlet_and_silo_distributions.png"), use_container_width=True)
            st.caption("Comparison of Extreme Silo partitions vs. Dirichlet continuous label skew (alpha = 0.5 and alpha = 0.1).")
        
    with tab5:
        st.subheader("Communication Overhead and Quantization Compression")
        if (CHARTS_DIR / "quantization_compression.png").exists():
            st.image(str(CHARTS_DIR / "quantization_compression.png"), use_container_width=True)
        st.image(str(CHARTS_DIR / "communication_cost.png"), use_container_width=True)
        
        st.markdown("#### Cumulative Communication Overhead (KB)")
        comm_df = pd.read_csv(EXPERIMENTS_DIR / "communication_overhead.csv")
        st.dataframe(comm_df, use_container_width=True, hide_index=True)

    with tab6:
        st.subheader("Byzantine Robustness against Poisoning (Label Flipping)")
        if (CHARTS_DIR / "poisoning_robustness.png").exists():
            st.image(str(CHARTS_DIR / "poisoning_robustness.png"), use_container_width=True)
        st.caption("Coordinate Median and Trimmed Mean robust aggregators reject poisoned client updates.")

    with tab7:
        st.subheader("Differential Privacy Budget Tradeoff (Epsilon vs. F1 & Recall)")
        if (CHARTS_DIR / "dp_epsilon_vs_f1.png").exists():
            st.image(str(CHARTS_DIR / "dp_epsilon_vs_f1.png"), use_container_width=True)
        st.caption("Evaluated at delta = 1e-5 across noise multipliers sigma in {0.5, 1.0, 2.0}.")

# PAGE 5: SYSTEM ARCHITECTURE
elif nav == "System Architecture":
    st.title("System Architecture and Threat Model")
    st.markdown("""
    ```
                    ┌──────────────────────────────────────────────┐
                    │          Flower FedAvg Server                │
                    │        Global IDS Model (MLP)                │
                    └───────┬──────────────┬──────────────┬────────┘
                            │              │              │
             Global Weights │              │              │ Global Weights
                   (76.8 KB)│              │              │ (76.8 KB)
                            ▼              ▼              ▼
                     ┌──────────────┐┌──────────────┐┌──────────────┐
                     │   Client A   ││   Client B   ││   Client C   │
                     │  (DoS Silo)  ││ (PortScan)   ││(Web Attacks) │
                     └──────┬───────┘└──────┬───────┘└──────┬───────┘
                            │               │               │
                     Local SGD (2 ep)Local SGD (2 ep)Local SGD (2 ep)
                     [PRIVATE DATA]  [PRIVATE DATA]  [PRIVATE DATA]
                     (Never leaves)  (Never leaves)  (Never leaves)
    ```
    """)
    st.markdown("""
    ### Privacy and Security Guarantees:
    - **Raw Data Locality:** Network flows, IP addresses, and timestamps remain strictly inside the organizational boundary.
    - **Flower Orchestration:** Clients download global parameter vectors theta_t, perform local training updates delta_theta_k, and upload updated vectors.
    - **Residual Privacy Risks:** Model inversion and membership inference attacks remain theoretical risks if an adversary inspects raw weight differentials.
    """)
