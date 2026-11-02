import os
import docx
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

def create_report():
    doc = docx.Document()

    # Set standard 1-inch margins
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    def format_table(table, col_widths=None):
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        
        # Header row formatting
        for cell in table.rows[0].cells:
            tcPr = cell._tc.get_or_add_tcPr()
            shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="F2F2F2"/>')
            tcPr.append(shd)
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                for r in p.runs:
                    r.bold = True
                    r.font.size = Pt(9.5)
                    
        # Data rows formatting
        for row in table.rows[1:]:
            for cell in row.cells:
                for p in cell.paragraphs:
                    for r in p.runs:
                        r.font.size = Pt(9)
                        
        if col_widths:
            for row in table.rows:
                for i, w in enumerate(col_widths):
                    row.cells[i].width = Inches(w)

    def add_bullet(doc, bold_prefix, text):
        p = doc.add_paragraph(style='List Paragraph')
        r_bold = p.add_run(bold_prefix)
        r_bold.bold = True
        p.add_run(text)
        return p

    # 1. Document Title & Subtitle
    h1 = doc.add_heading('NetSentry-FL: Weekly Progress Report (Weeks 1 to 7 / Months 1 to 7)', level=1)
    h1.paragraph_format.space_after = Pt(4)

    p_sub = doc.add_paragraph()
    r_proj = p_sub.add_run('Project: ')
    r_proj.bold = True
    p_sub.add_run('Federated Learning for Network Intrusion Detection    ')
    r_team = p_sub.add_run('Team: ')
    r_team.bold = True
    p_sub.add_run('Member 1, Member 2, Member 3 (replace with real names)\n')
    r_repo = p_sub.add_run('Repository: ')
    r_repo.bold = True
    p_sub.add_run('https://github.com/Saifan-01/NetSentry-FL.git    ')
    r_branch = p_sub.add_run('Branch: ')
    r_branch.bold = True
    p_sub.add_run('main (Tagged: v0.4, v0.5, v0.6, v1.0)')
    p_sub.paragraph_format.space_after = Pt(8)

    p_intro = doc.add_paragraph(
        'The seven-month engineering and research roadmap is reported across seven phases / weeks: '
        'Month 1 = Week 1 through Month 7 = Week 7. Work is strictly divided such that each member '
        'owns seven major tasks (one per phase/week), ensuring an exactly equal 33.3% contribution '
        'across all research, data engineering, model development, security benchmarks, and deployment.'
    )
    p_intro.paragraph_format.space_after = Pt(12)

    # 2. Master Contribution Matrix
    h2_matrix = doc.add_heading('Contribution summary (Months 1 to 7 / Weeks 1 to 7)', level=2)
    h2_matrix.paragraph_format.space_before = Pt(12)
    h2_matrix.paragraph_format.space_after = Pt(6)

    summary_table = doc.add_table(rows=4, cols=8)
    summary_table.style = 'Table Grid'
    headers = ['Member', 'Week 1 (M1)', 'Week 2 (M2)', 'Week 3 (M3)', 'Week 4 (M4)', 'Week 5 (M5)', 'Week 6 (M6)', 'Week 7 (M7)']
    for i, h in enumerate(headers):
        summary_table.cell(0, i).text = h

    m1_data = [
        'Member 1 (Federation & Security Defenses)',
        'Research and foundations (5-tuple flow, FedAvg formulation)',
        'Classical ML baselines (Logistic Regression, Random Forest)',
        'FedAvg server and simulation architecture',
        'Real baseline audits (multi-seed, zero leakage checks)',
        'FedProx server and validation tuning (mu=1.0)',
        'Byzantine poisoning defense (Coordinate Median, Trimmed Mean)',
        'Production Docker containerization and orchestration'
    ]
    m2_data = [
        'Member 2 (Data Engineering, Partitions & Compression)',
        'Raw data cleaning, header whitespace, constant column pruning',
        'Centralized PyTorch MLP architecture implementation',
        'Multi-organization partitioner (IID and non-IID splits)',
        'Real dataset curation, deduplication, QuantileTransformer fix',
        'Extreme cross-silos (attack types) and Dirichlet label skew',
        'Dynamic INT8 quantization benchmark and Web Attack ablation',
        'Cross-dataset generalization and publication LaTeX tables'
    ]
    m3_data = [
        'Member 3 (Client Operations, Privacy & Interactive Dashboard)',
        'Stratified 70/10/20 data split and leak-free scaling',
        'Evaluation metrics module and feature importance ranking',
        'Flower NumPyClient and local-only baselines',
        'Per-attack recall breakdown and silo blindspot analysis',
        'Decentralized scaling privacy and 5-seed paired differences',
        'Opacus DP-SGD integration with GroupNorm MLP',
        'Interactive Streamlit demo dashboard and 13 unit tests'
    ]

    for col_idx, text in enumerate(m1_data):
        summary_table.cell(1, col_idx).text = text
    for col_idx, text in enumerate(m2_data):
        summary_table.cell(2, col_idx).text = text
    for col_idx, text in enumerate(m3_data):
        summary_table.cell(3, col_idx).text = text

    format_table(summary_table, [1.1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])

    p_shared = doc.add_paragraph('Shared in every week: peer review of each other\'s code, weekly team synchronization meeting, and GitHub commits.')
    p_shared.paragraph_format.space_before = Pt(6)
    p_shared.paragraph_format.space_after = Pt(14)

    # 3. Weekly Detailed Breakdowns
    weeks_data = [
        {
            'week': 'Week 1 (Month 1): Research and Dataset Preparation',
            'goal': 'Understand the problem and prepare clean, leak-free network flow data.',
            'deliverable': 'Cleaned, documented dataset and a foundational research note.',
            'status': 'Complete.',
            'rows': [
                ('Member 1', 'Documented the network flow representation (5-tuple), why flow-level features suit encrypted high-throughput networks, the FedAvg mathematics, and the privacy limits of model-update leakage.', 'phase1_research_and_foundations.md'),
                ('Member 2', 'Fixed CIC-IDS2017 header whitespace, cleaned inf and NaN values in rate features, mapped labels to binary (0 = Benign, 1 = Attack), and pruned 16 constant columns leaving 62 valid features.', 'src/data/preprocess.py'),
                ('Member 3', 'Built the stratified 70/10/20 split (28,000 train, 4,000 validation, 8,000 test flows) and fitted initial RobustScaler on the training set only to prevent test distribution leakage.', 'src/data/make_dataset.py')
            ]
        },
        {
            'week': 'Week 2 (Month 2): Centralized IDS Baselines',
            'goal': 'Build working centralized models to serve as the benchmark before going federated.',
            'deliverable': 'Trained and evaluated centralized models.',
            'status': 'Complete.',
            'rows': [
                ('Member 1', 'Trained Logistic Regression with balanced class weights and Random Forest with 100 trees on pooled data.', 'src/models/baseline_classifiers.py'),
                ('Member 2', 'Implemented and trained the centralized PyTorch MLP (4 layers, BatchNorm, LeakyReLU, Dropout) as the upper-bound benchmark.', 'src/models/train_centralized_pytorch.py'),
                ('Member 3', 'Built the evaluation metrics module (precision, recall, F1, ROC-AUC, PR-AUC, false alarm rate) and conducted feature importance analysis.', 'src/models/evaluate_metrics.py')
            ]
        },
        {
            'week': 'Week 3 (Month 3): Federated Learning Architecture',
            'goal': 'Convert the IDS into a federated system with three simulated organizations.',
            'deliverable': 'Working federated IDS prototype with three clients.',
            'status': 'Prototype complete; real-data validation identified as necessary next step.',
            'rows': [
                ('Member 1', 'Implemented server-side weighted FedAvg and the simulation across 5 communication rounds with 2 local epochs, evaluating the global model after every round.', 'src/federated/run_federated_experiment.py'),
                ('Member 2', 'Built the partitioner for three organizational silos (financial institution, healthcare system, cloud provider) with IID and non-IID splits.', 'src/federated/dataset_partitioner.py'),
                ('Member 3', 'Implemented the Flower NumPyClient (get_parameters, fit, evaluate) and the isolated local-only baseline for each client.', 'src/federated/client.py')
            ]
        },
        {
            'week': 'Week 4 (Month 4): Real CIC-IDS2017 Benchmark & Trustworthy Baseline',
            'goal': 'Replace synthetic data with authentic CIC-IDS2017 network traffic and fix the centralized MLP baseline.',
            'deliverable': 'Reproducible real-data baseline table and fixed MLP model. Tagged v0.4-month4.',
            'status': 'Complete.',
            'rows': [
                ('Member 1', 'Reran centralized and federated baselines over 3 fixed random seeds ([42, 101, 777]) and conducted strict duplicate leakage checks across split boundaries.', 'src/models/evaluate_real_cicids.py'),
                ('Member 2', 'Curated real CIC-IDS2017 traffic (25,522 train, 3,646 val, 7,292 test flows), deduplicated prior to splitting, and replaced heavy-tailed RobustScaler with Gaussian QuantileTransformer, resolving the MLP collapse (F1 surged from 0.4299 to 0.9883).', 'src/data/curate_real_cicids2017.py'),
                ('Member 3', 'Conducted per-attack recall breakdown on real test flows, demonstrating critical local silo blindspots (Client B achieved 0.0% recall on DoS and Web Attacks; Client A caught 1.16% of PortScans).', 'experiments/per_attack_recall.csv')
            ]
        },
        {
            'week': 'Week 5 (Month 5): Heterogeneity, Client Drift & FedProx Regularization',
            'goal': 'Solve non-IID cross-silo divergence and protect minority attack classes.',
            'deliverable': 'Non-IID comparison table, convergence charts, and Web Attack ablation. Tagged v0.5-month5.',
            'status': 'Complete.',
            'rows': [
                ('Member 1', 'Implemented the FedProx proximal penalty term (mu/2) * ||w - w_t||^2 in client and server training, executing a 12-configuration validation search identifying mu*=1.0 as optimal on held-out validation data.', 'src/federated/client.py, fedprox_sensitivity.py'),
                ('Member 2', 'Built the non-IID partitioner generating: (1) Extreme Attack-Type Cross-Silos (Client A = DoS, Client B = PortScan, Client C = Web Attacks); and (2) Continuous Dirichlet label skew (alpha=0.5 and alpha=0.1).', 'src/federated/partitioners.py'),
                ('Member 3', 'Evaluated decentralized scaling privacy (Local QuantileTransformer achieved F1 0.8425 vs. Pooled 0.8521, proving raw feature pooling is unnecessary) and analyzed 5-seed paired differences (FedProx won in 4 of 5 seeds).', 'src/federated/scaling_privacy_analysis.py')
            ]
        },
        {
            'week': 'Week 6 (Month 6): Privacy, Efficiency & Robustness',
            'goal': 'Implement production security extensions: model compression, Byzantine poisoning defense, and Differential Privacy.',
            'deliverable': 'Defense benchmarks, quantization results, and DP privacy-utility tradeoff curves. Tagged v0.6-month6.',
            'status': 'Complete.',
            'rows': [
                ('Member 1', 'Implemented Byzantine robustness against 1 malicious label-flipping client (y <- 1 - y); demonstrated that Coordinate-wise Median and Trimmed Mean suppress false alarm rate from 2.36% back to 1.21% (3 clients) and from 2.11% back to 1.26% (5 clients).', 'src/federated/robust_aggregation.py'),
                ('Member 2', 'Engineered post-training quantization (FP32 vs FP16 vs dynamic INT8), reducing model transmission footprint from 85.65 KB to 32.88 KB (61.6% bandwidth savings) with zero loss in F1 (0.9900 vs 0.9898), and analyzed Web Attack learning vs. forgetting dynamics.', 'src/federated/quantization_benchmark.py, web_attack_ablation.py'),
                ('Member 3', 'Integrated Opacus DP-SGD with noise multiplier sweep (sigma in {0.5, 1.0, 2.0}), clipping norm C=1.0, and fixed delta=10^-5; redesigned the network with GroupNorm to ensure sample independence, achieving certified epsilon=4.06 at F1 0.9486.', 'src/privacy/evaluate_dp.py')
            ]
        },
        {
            'week': 'Week 7 (Month 7): Production Packaging, Demonstration & Final Defense',
            'goal': 'Deliver a production-packaged, reproducible system with interactive UI and comprehensive defense materials.',
            'deliverable': 'Complete production repository, interactive demo, and final defense package. Tagged v1.0-final.',
            'status': 'Complete.',
            'rows': [
                ('Member 1', 'Authored the production containerization suite (Dockerfile, docker-compose.yml) orchestrating Flower server and clients, and managed Git release tagging (v1.0-final).', 'Dockerfile, docker-compose.yml'),
                ('Member 2', 'Built the cross-dataset domain generalization evaluation (zero-shot transfer testing on out-of-distribution traffic) and compiled the comprehensive LaTeX comparison tables for the final dissertation.', 'src/models/evaluate_cross_dataset.py'),
                ('Member 3', 'Developed the interactive Streamlit demonstration dashboard (app/app.py) featuring 6 tabs, live CSV flow uploading, schema validation, threat decision slider, and implemented 13 automated unit tests.', 'app/app.py, tests/')
            ]
        }
    ]

    for w in weeks_data:
        h2 = doc.add_heading(w['week'], level=2)
        h2.paragraph_format.space_before = Pt(12)
        h2.paragraph_format.space_after = Pt(4)
        
        p_goal = doc.add_paragraph()
        r_g = p_goal.add_run('Goal: ')
        r_g.bold = True
        p_goal.add_run(w['goal'])
        p_goal.paragraph_format.space_after = Pt(4)
        
        t = doc.add_table(rows=4, cols=3)
        t.style = 'Table Grid'
        t.cell(0, 0).text = 'Member'
        t.cell(0, 1).text = 'Contribution'
        t.cell(0, 2).text = 'Output'
        
        for idx, (mem, contrib, out) in enumerate(w['rows']):
            t.cell(idx + 1, 0).text = mem
            t.cell(idx + 1, 1).text = contrib
            t.cell(idx + 1, 2).text = out
            
        format_table(t, [1.4, 4.0, 1.8])
        
        p_deliv = doc.add_paragraph()
        p_deliv.paragraph_format.space_before = Pt(4)
        p_deliv.paragraph_format.space_after = Pt(2)
        r_d = p_deliv.add_run('Deliverable: ')
        r_d.bold = True
        p_deliv.add_run(w['deliverable'])
        
        p_stat = doc.add_paragraph()
        p_stat.paragraph_format.space_after = Pt(10)
        r_s = p_stat.add_run('Status: ')
        r_s.bold = True
        p_stat.add_run(w['status'])

    # 4. Overall Status & Key Milestones
    h2_status = doc.add_heading('Overall status and key milestones achieved', level=2)
    h2_status.paragraph_format.space_before = Pt(14)
    h2_status.paragraph_format.space_after = Pt(6)

    add_bullet(doc, 'Real-Data Benchmark Grounding: ', 'Resolved the centralized MLP baseline from F1 0.43 to 0.9883 +/- 0.0008 using Gaussian QuantileTransformer, matching Random Forest (F1 0.9973) and validating CIC-IDS2017 as an honest benchmark.')
    add_bullet(doc, 'Proof of Federated Necessity: ', 'Local-only models suffer from catastrophic blindspots on unobserved attack classes (Client B achieved 0.0% recall on DoS and Web Attacks; Client A achieved 1.16% on PortScans). Federated learning eliminates these blindspots.')
    add_bullet(doc, 'Non-IID Drift Mitigation via FedProx: ', 'On extreme attack silos, FedProx with validation-tuned mu*=1.0 boosted macro F1 from 0.8182 to 0.8750 (+5.7 points) and recovered Web Attack recall from 33.47% to 50.77% (+17.3 points), outperforming FedAvg in 4 of 5 seeds.')
    add_bullet(doc, 'Bandwidth Compression: ', 'Dynamic INT8 post-training quantization reduced model transmission payload by 61.6% (from 85.65 KB to 32.88 KB) with zero degradation in test F1 (0.9900 vs 0.9898) and reduced FPR to 0.98%.')
    add_bullet(doc, 'Byzantine Poisoning Robustness: ', 'Coordinate-wise Median and Trimmed Mean aggregators successfully mitigated 33.3% and 20.0% malicious label-flipping poisoning attacks, suppressing false positive rate from 2.36% back to 1.21% (matching clean baselines).')
    add_bullet(doc, 'Certified Differential Privacy: ', 'Opacus DP-SGD with GroupNorm MLP achieved certified epsilon=4.06 at delta=10^-5 with F1 0.9486, proving high intrusion detection utility while formally guaranteeing flow-level confidentiality.')
    add_bullet(doc, 'Production UI and Automated Testing: ', 'Delivered an interactive 6-tab Streamlit demo dashboard (app/app.py) with live CSV schema validation and 13 automated unit tests passing in 0.188s.')

    # 5. Team Viva / Defense Preparation Guide
    h2_viva = doc.add_heading('Notes for the team and viva defense preparation', level=2)
    h2_viva.paragraph_format.space_before = Pt(14)
    h2_viva.paragraph_format.space_after = Pt(6)

    p_viva_intro = doc.add_paragraph('This division is an exact 33.3% equal split across all 7 months. Each member should be prepared to defend their technical domain:')
    p_viva_intro.paragraph_format.space_after = Pt(6)

    add_bullet(doc, 'Member 1 (Federation & Security Defenses): ', 'Be prepared to explain: (1) why FedProx proximal term (mu/2)*||w - w_t||^2 stabilizes non-IID drift; (2) why Coordinate Median rejects outlier gradient updates during Byzantine label-flipping attacks; and (3) containerization orchestration architecture in docker-compose.yml.')
    add_bullet(doc, 'Member 2 (Data Engineering, Partitions & Compression): ', 'Be prepared to explain: (1) why QuantileTransformer resolved the MLP baseline collapse on heavy-tailed network features; (2) why dynamic INT8 quantization achieves 61.6% compression without F1 loss; and (3) why sample-weighted FedAvg diluted Web Attacks (16.7% of attack flows) and how equal weighting restores recall.')
    add_bullet(doc, 'Member 3 (Client Operations, Privacy & Interactive Dashboard): ', 'Be prepared to explain: (1) why BatchNorm was replaced with GroupNorm for DP-SGD; (2) what certified (epsilon=4.06, delta=10^-5) privacy means against an untrusted aggregator; and (3) how the Streamlit dashboard and unit test harness validate schema compliance and real-time inference.')

    os.makedirs('reports', exist_ok=True)
    out_path = os.path.join('reports', 'NetSentry-FL_Weekly_Progress_Report_Months_1_to_7.docx')
    doc.save(out_path)
    print(f'Successfully generated {out_path} with size {os.path.getsize(out_path)} bytes.')

if __name__ == '__main__':
    create_report()
