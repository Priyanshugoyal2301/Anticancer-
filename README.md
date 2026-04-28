# 🧬 Anticancer GNN — Full Research Pipeline

Graph Neural Network for predicting anticancer activity from molecular SMILES strings.  
Trains under **4 conditions** × **5-fold CV** and generates **44 visualizations**.

---

## Project Structure

```
.
├── anticancer_gnn_full.py    ← Main training script
├── app.py                    ← Streamlit deployment app
├── requirements.txt
├── Anticancer_dataset (1).csv
├── saved_models/             ← Checkpoints (auto-created)
│   ├── Random_NoAug_best.pt
│   ├── Random_Aug_best.pt
│   ├── Scaffold_NoAug_best.pt
│   ├── Scaffold_Aug_best.pt
│   └── meta.json
├── plots/                    ← 44 plots (auto-created)
└── results_summary.csv       ← Final metric table
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt

# For torch-geometric on GPU (adjust CUDA version):
pip install torch-scatter torch-sparse \
  -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### 2. Run Training

```bash
python anticancer_gnn_full.py
```

Training runs **4 conditions × 5 folds** sequentially.  
On GPU (T4): ~30–60 min total.  
On CPU: 3–5 hours (reduce `EPOCHS` to 80 for quick test).

**Output:**
- `saved_models/` — best model per condition
- `plots/` — 44 publication-quality plots
- `results_summary.csv` — final metric table

### 3. Launch Web App

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## 🧠 Model Architecture

**Multi-Scale Message Passing Network (MSMP-GNN)**

Three parallel GNN branches capture complementary structural patterns:

| Branch | Layer | Edge attr? |
|--------|-------|-----------|
| GCN    | `GCNConv × 2`        | ✗ |
| GATv2  | `GATv2Conv × 2`      | ✓ |
| GT     | `TransformerConv × 2`| ✓ |

All branches use **residual connections + LayerNorm + GELU**.

**Readout:** mean + max pooling → concatenate with:
- Morgan fingerprint (1024-bit, radius=2)
- 9 molecular descriptors (MolWt, LogP, TPSA, …, QED)

**Two separate heads:**
- **Classification head** (BCE loss) — Active/Inactive
- **Regression head** (SmoothL1 loss, deeper with SiLU) — pIC50

---

## 📊 Training Conditions

| Condition | Split | Augmentation |
|-----------|-------|-------------|
| Random \| No Aug   | KFold random   | ✗ |
| Random \| Aug      | KFold random   | ✓ |
| Scaffold \| No Aug | Murcko scaffold| ✗ |
| Scaffold \| Aug    | Murcko scaffold| ✓ |

**Augmentation strategies (applied at 50/25/25 ratio):**
1. SMILES randomisation (non-canonical → new graph topology)
2. Edge dropout (15%)
3. Node feature noise (σ=0.05)

---

## 📈 Regression Improvements

Key changes over previous version:

| Issue | Fix |
|-------|-----|
| Shared fc → regression head too shallow | Separate 4-layer regression head |
| ReLU in regression head | SiLU (smoother gradients) |
| loss = 0.6 cls + 0.9 reg | loss = 0.5 cls + 1.2 reg (regression prioritized) |
| Single GNN layer | 2 layers per branch (deeper) |
| BatchNorm in GNN | LayerNorm (handles variable-size graphs) |
| Constant LR | OneCycleLR scheduler |

---

## 🌐 Deployment Notes

The Streamlit app (`app.py`) supports:
- Single SMILES input
- Batch prediction (one SMILES per line)
- Model selection (any of the 4 trained conditions)
- Adjustable classification threshold
- CSV download of results
- Molecule 2D structure rendering

To deploy on a server:

```bash
# Production with gunicorn-like config:
streamlit run app.py \
  --server.port 8080 \
  --server.address 0.0.0.0 \
  --server.headless true
```

For cloud deployment (Hugging Face Spaces, Streamlit Cloud):
- Upload `app.py`, `requirements.txt`, and `saved_models/` folder
- Set Python ≥ 3.10

---

## 📋 Plots Generated (44 total)

| Section | Count | Description |
|---------|-------|-------------|
| Chemical Space | 2 | Tanimoto distribution, pIC50 histogram |
| Classification Bars | 8 | ACC, AUC, MCC, Sens, Spec, F1, Prec, BAcc |
| ROC Curves | 5 | 4 individual + 1 combined |
| Regression Scatter | 4 | Predicted vs True per condition |
| Confusion Matrices | 4 | Per condition |
| Violin Plots | 2 | ACC + AUC across folds |
| Augmentation Effect | 2 | Per split type |
| Random vs Scaffold | 2 | No aug + With aug |
| R² Heatmap | 1 | All folds × conditions |
| Sim vs Accuracy | 1 | Tanimoto vs fold accuracy |
| Convergence | 4 | Loss curves per condition |
| Residuals | 8 | Scatter + histogram per condition |
| t-SNE | 2 | pIC50 coloured + Activity coloured |
| PR Curves | 5 | 4 individual + 1 combined |
| Aug Gain Summary | 1 | Random vs Scaffold gain bar |
