"""
Anticancer GNN — Streamlit Web App v2
=======================================
4 Models (GCN | GAT | GT | MSMP) × 4 Conditions
Predicts anticancer activity (classification + pIC50 regression) from SMILES.

Run:
    pip install streamlit torch torch-geometric rdkit-pypi
    streamlit run app.py
"""

import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, QED, Draw
RDLogger.DisableLog("rdApp.*")

from torch_geometric.data import Data, Batch
from torch_geometric.nn import (
    GCNConv, GATv2Conv, TransformerConv,
    global_mean_pool, global_max_pool,
)

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
MODEL_DIR = "saved_models"

MODEL_NAMES = ["GCN", "GAT", "GT", "MSMP"]
COND_NAMES  = ["Random_NoAug", "Random_Aug", "Scaffold_NoAug", "Scaffold_Aug"]
COND_LABELS = {
    "Random_NoAug"   : "Random | No Aug",
    "Random_Aug"     : "Random | With Aug",
    "Scaffold_NoAug" : "Scaffold | No Aug",
    "Scaffold_Aug"   : "Scaffold | With Aug",
}
MODEL_COLORS = {
    "GCN" : "#4C72B0",
    "GAT" : "#DD8452",
    "GT"  : "#55A868",
    "MSMP": "#C44E52",
}

# ─────────────────────────────────────────────────────────────
# Model definitions  (must match training script exactly)
# ─────────────────────────────────────────────────────────────

class RegHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,512), nn.BatchNorm1d(512), nn.SiLU(), nn.Dropout(0.3),
            nn.Linear(512,256),    nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(256,128),    nn.SiLU(),
            nn.Linear(128,1),
        )
    def forward(self, x): return self.net(x)

class ClsHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512,256),    nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256,1),
        )
    def forward(self, x): return self.net(x)

class GNNBlock(nn.Module):
    def __init__(self, conv, hidden, use_edge_attr=True):
        super().__init__()
        self.conv = conv; self.use_edge_attr = use_edge_attr
        self.norm = nn.LayerNorm(hidden); self.drop = nn.Dropout(0.1)
    def forward(self, x, ei, ea=None):
        h = (self.conv(x,ei,ea) if self.use_edge_attr and ea is not None
             else self.conv(x,ei))
        return self.drop(F.gelu(self.norm(h)))

class GCNModel(nn.Module):
    def __init__(self, node_dim, edge_dim):
        super().__init__()
        H=128
        self.conv1=GNNBlock(GCNConv(node_dim,H),H,use_edge_attr=False)
        self.conv2=GNNBlock(GCNConv(H,H),H,use_edge_attr=False)
        fuse=H*2+1024+9; self.norm=nn.LayerNorm(fuse)
        self.cls=ClsHead(fuse); self.reg=RegHead(fuse)
    def forward(self, data):
        x=self.conv2(self.conv1(data.x,data.edge_index),data.edge_index)
        p=torch.cat([global_mean_pool(x,data.batch),global_max_pool(x,data.batch)],dim=1)
        f=self.norm(torch.cat([p,data.fp.view(p.size(0),-1),data.desc.view(p.size(0),-1)],dim=1))
        return self.cls(f).squeeze(-1), self.reg(f).squeeze(-1)

class GATModel(nn.Module):
    def __init__(self, node_dim, edge_dim):
        super().__init__()
        H=128
        self.conv1=GNNBlock(GATv2Conv(node_dim,H,heads=4,concat=False,edge_dim=edge_dim),H)
        self.conv2=GNNBlock(GATv2Conv(H,H,heads=4,concat=False,edge_dim=edge_dim),H)
        fuse=H*2+1024+9; self.norm=nn.LayerNorm(fuse)
        self.cls=ClsHead(fuse); self.reg=RegHead(fuse)
    def forward(self, data):
        ea=data.edge_attr
        x=self.conv2(self.conv1(data.x,data.edge_index,ea),data.edge_index,ea)
        p=torch.cat([global_mean_pool(x,data.batch),global_max_pool(x,data.batch)],dim=1)
        f=self.norm(torch.cat([p,data.fp.view(p.size(0),-1),data.desc.view(p.size(0),-1)],dim=1))
        return self.cls(f).squeeze(-1), self.reg(f).squeeze(-1)

class GTModel(nn.Module):
    def __init__(self, node_dim, edge_dim):
        super().__init__()
        H=128
        self.conv1=GNNBlock(TransformerConv(node_dim,H,heads=4,concat=False,edge_dim=edge_dim),H)
        self.conv2=GNNBlock(TransformerConv(H,H,heads=4,concat=False,edge_dim=edge_dim),H)
        fuse=H*2+1024+9; self.norm=nn.LayerNorm(fuse)
        self.cls=ClsHead(fuse); self.reg=RegHead(fuse)
    def forward(self, data):
        ea=data.edge_attr
        x=self.conv2(self.conv1(data.x,data.edge_index,ea),data.edge_index,ea)
        p=torch.cat([global_mean_pool(x,data.batch),global_max_pool(x,data.batch)],dim=1)
        f=self.norm(torch.cat([p,data.fp.view(p.size(0),-1),data.desc.view(p.size(0),-1)],dim=1))
        return self.cls(f).squeeze(-1), self.reg(f).squeeze(-1)

class MSMPModel(nn.Module):
    def __init__(self, node_dim, edge_dim):
        super().__init__()
        H=128
        self.gcn1=GNNBlock(GCNConv(node_dim,H),H,use_edge_attr=False)
        self.gcn2=GNNBlock(GCNConv(H,H),H,use_edge_attr=False)
        self.gat1=GNNBlock(GATv2Conv(node_dim,H,heads=4,concat=False,edge_dim=edge_dim),H)
        self.gat2=GNNBlock(GATv2Conv(H,H,heads=4,concat=False,edge_dim=edge_dim),H)
        self.gt1 =GNNBlock(TransformerConv(node_dim,H,heads=4,concat=False,edge_dim=edge_dim),H)
        self.gt2 =GNNBlock(TransformerConv(H,H,heads=4,concat=False,edge_dim=edge_dim),H)
        fuse=H*3*2+1024+9; self.norm=nn.LayerNorm(fuse)
        self.cls=ClsHead(fuse); self.reg=RegHead(fuse)
    def forward(self, data):
        x,ei,ea,b=data.x,data.edge_index,data.edge_attr,data.batch
        g1=self.gcn2(self.gcn1(x,ei),ei)
        g2=self.gat2(self.gat1(x,ei,ea),ei,ea)
        g3=self.gt2(self.gt1(x,ei,ea),ei,ea)
        def pool(h):
            return torch.cat([global_mean_pool(h,b),global_max_pool(h,b)],dim=1)
        p=torch.cat([pool(g1),pool(g2),pool(g3)],dim=1)
        f=self.norm(torch.cat([p,data.fp.view(p.size(0),-1),data.desc.view(p.size(0),-1)],dim=1))
        return self.cls(f).squeeze(-1), self.reg(f).squeeze(-1)

MODEL_CLASSES = {"GCN":GCNModel,"GAT":GATModel,"GT":GTModel,"MSMP":MSMPModel}

# ─────────────────────────────────────────────────────────────
# Feature extraction  (identical to training)
# ─────────────────────────────────────────────────────────────

def atom_features(atom):
    hyb = atom.GetHybridization()
    try:
        c = float(atom.GetProp("_GasteigerCharge"))
        if np.isnan(c) or np.isinf(c): c = 0.0
    except: c = 0.0
    return [
        atom.GetAtomicNum()/100, atom.GetDegree()/8,
        atom.GetFormalCharge()/4, atom.GetTotalNumHs()/8,
        atom.GetTotalValence()/8, int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
        int(hyb==Chem.rdchem.HybridizationType.SP),
        int(hyb==Chem.rdchem.HybridizationType.SP2),
        int(hyb==Chem.rdchem.HybridizationType.SP3), c,
    ]

def bond_features(bond):
    bt = bond.GetBondType()
    return [
        float(bt==Chem.rdchem.BondType.SINGLE),
        float(bt==Chem.rdchem.BondType.DOUBLE),
        float(bt==Chem.rdchem.BondType.TRIPLE),
        float(bt==Chem.rdchem.BondType.AROMATIC),
        float(bond.GetIsConjugated()), float(bond.IsInRing()),
    ]

def mol_descriptors(mol):
    return [
        Descriptors.MolWt(mol)/1000, Descriptors.MolLogP(mol)/10,
        Descriptors.TPSA(mol)/150,   Descriptors.NumRotatableBonds(mol)/20,
        Descriptors.NumHAcceptors(mol)/10, Descriptors.NumHDonors(mol)/10,
        Descriptors.NumAromaticRings(mol)/10, Descriptors.NumHeteroatoms(mol)/20,
        QED.qed(mol),
    ]

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    mol = Chem.AddHs(mol)
    try: AllChem.ComputeGasteigerCharges(mol)
    except: pass

    x, edges, eattr = [], [], []
    for atom in mol.GetAtoms(): x.append(atom_features(atom))
    for bond in mol.GetBonds():
        i,j=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
        bf=bond_features(bond)
        edges+=[[i,j],[j,i]]; eattr+=[bf,bf]

    fp   = np.array(AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024),dtype=np.float32)
    desc = np.array(mol_descriptors(mol), dtype=np.float32)

    ei = torch.tensor(np.array(edges).T,dtype=torch.long) if edges else torch.zeros((2,0),dtype=torch.long)
    ea = torch.tensor(np.array(eattr), dtype=torch.float) if eattr else torch.zeros((0,6),dtype=torch.float)

    return Data(
        x          = torch.tensor(x,    dtype=torch.float),
        edge_index = ei, edge_attr = ea,
        fp         = torch.tensor(fp,   dtype=torch.float).unsqueeze(0),
        desc       = torch.tensor(desc, dtype=torch.float).unsqueeze(0),
        y_cls      = torch.tensor([0.], dtype=torch.float),
        y_reg      = torch.tensor([0.], dtype=torch.float),
    )

# ─────────────────────────────────────────────────────────────
# Cached loaders
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_meta():
    path = os.path.join(MODEL_DIR, "meta.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

@st.cache_resource
def load_single_model(model_name, cond_name):
    path = os.path.join(MODEL_DIR, f"{model_name}_{cond_name}_best.pt")
    if not os.path.exists(path):
        return None, None
    ckpt  = torch.load(path, map_location="cpu")
    model = MODEL_CLASSES[model_name](ckpt["node_dim"], ckpt["edge_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt

@st.cache_resource
def load_all_models():
    """Load all 16 model checkpoints at once."""
    meta = load_meta()
    loaded = {}
    for mname in MODEL_NAMES:
        loaded[mname] = {}
        for cond in COND_NAMES:
            model, ckpt = load_single_model(mname, cond)
            loaded[mname][cond] = model
    return loaded, meta

# ─────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, graphs_batch, Y_MEAN, Y_STD, threshold=0.5):
    batch = Batch.from_data_list(graphs_batch)
    cls_out, reg_out = model(batch)
    probs  = torch.sigmoid(cls_out).numpy().flatten()
    pic50s = (reg_out.numpy().flatten() * Y_STD + Y_MEAN)
    preds  = (probs >= threshold).astype(int)
    return probs, pic50s, preds

# ─────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Anticancer GNN Predictor",
    page_icon="🧬",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .main-title   {font-size:2.3rem;font-weight:800;color:#1a3a6b;margin-bottom:0}
    .sub-title    {font-size:1rem;color:#555;margin-bottom:1.5rem}
    .model-badge  {display:inline-block;padding:0.25rem 0.75rem;border-radius:12px;
                   font-weight:700;font-size:0.85rem;margin:2px}
    .active-box   {background:#d4f4dd;color:#145a32;border-radius:10px;
                   padding:0.6rem 1rem;font-weight:700;font-size:1.1rem}
    .inactive-box {background:#fde8e8;color:#7b241c;border-radius:10px;
                   padding:0.6rem 1rem;font-weight:700;font-size:1.1rem}
    .metric-box   {background:#f0f4ff;border-radius:10px;padding:0.8rem 1rem;
                   border-left:4px solid #1a3a6b;margin-bottom:0.5rem}
    .stDataFrame  {font-size:0.88rem}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🧬 Anticancer Activity Predictor</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">GCN · GATv2 · Graph Transformer · MSMP  ·  '
    '4 Training Conditions  ·  Classification + pIC50 Regression</p>',
    unsafe_allow_html=True
)
st.divider()

# ─── Load meta & models ──────────────────────────────────────
meta = load_meta()
if meta is None:
    st.error(
        "⚠️  `saved_models/meta.json` not found.\n\n"
        "Please run `anticancer_gnn_full_v2.py` first to train all models."
    )
    st.stop()

Y_MEAN = meta["Y_MEAN"]; Y_STD = meta["Y_STD"]
THRESHOLD_PIC50 = meta.get("THRESHOLD", 7.0)

# ─── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️  Settings")

    app_mode = st.radio(
        "Prediction Mode",
        ["Single Model", "Compare All Models", "Compare All Conditions"],
        index=1,
    )

    cls_threshold = st.slider("Classification Threshold", 0.1, 0.9, 0.5, 0.01,
                              help="Probability cutoff for Active/Inactive")

    st.markdown("---")
    st.markdown("**Architecture Summary**")
    st.markdown("""
| Model | Architecture |
|-------|-------------|
| GCN   | GCNConv × 2 |
| GAT   | GATv2Conv × 2 (4 heads) |
| GT    | TransformerConv × 2 (4 heads) |
| MSMP  | GCN + GAT + GT in parallel |
""")
    st.markdown("---")
    st.markdown(f"**Active threshold:** pIC50 ≥ {THRESHOLD_PIC50}")
    st.markdown(f"**Y_MEAN:** {Y_MEAN:.3f}  |  **Y_STD:** {Y_STD:.3f}")

# ─── Example SMILES ──────────────────────────────────────────
EXAMPLES = {
    "Anticancer (Active)" :
        "CC1=C(N=NN1C2=CC=C(C=C2)F)/C(=N/NC3=NC(=CS3)C4=CC=CC=C4)C",
    "Moderate Activity"   :
        "COC1=CC2=C(C(=C1)OC)C(=O)NC(=N2)C3=CC=C(C=C3)O",
    "Low Activity"        :
        "CN(C)N1C(=O)C2=C3C4=C(C(=CC=C4)O)NC3=C5C(=C2C1=O)NC5=O",
    "Aspirin (Inactive)"  : "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine (Inactive)" : "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
}

# ─── Input Panel ─────────────────────────────────────────────
col_in, col_res = st.columns([1, 1.6], gap="large")

with col_in:
    st.subheader("🔬 Input")

    mode = st.radio("Input mode", ["Single SMILES", "Batch"], horizontal=True)

    if mode == "Single SMILES":
        ex_choice = st.selectbox("Load example", ["(none)"] + list(EXAMPLES.keys()))
        default   = EXAMPLES.get(ex_choice, "")
        smi_input = st.text_area("SMILES", value=default, height=80,
                                 placeholder="Paste SMILES here …")
        smiles_list = [smi_input.strip()] if smi_input.strip() else []
    else:
        smi_input = st.text_area("SMILES — one per line", height=180,
                                 placeholder="CC(=O)Oc1ccccc1C(=O)O\nCCO\n…")
        smiles_list = [s.strip() for s in smi_input.splitlines() if s.strip()]

    if app_mode == "Single Model":
        sel_model = st.selectbox("Model", MODEL_NAMES, index=3)
        sel_cond  = st.selectbox("Condition", COND_NAMES,
                                 format_func=lambda c: COND_LABELS[c])

    predict_btn = st.button("🚀 Predict", type="primary", use_container_width=True)

    # Molecule visualisation (single mode)
    if smiles_list and mode == "Single SMILES":
        mol_draw = Chem.MolFromSmiles(smiles_list[0])
        if mol_draw:
            img = Draw.MolToImage(mol_draw, size=(300, 220))
            st.image(img, caption="2D Structure", use_column_width=True)

# ─── Prediction & Results ────────────────────────────────────
with col_res:
    st.subheader("📊 Results")

    if predict_btn and smiles_list:

        # Build graph objects
        valid_graphs, valid_smiles = [], []
        for smi in smiles_list:
            g = smiles_to_graph(smi)
            if g is None: st.warning(f"Invalid SMILES skipped: `{smi}`")
            else:
                valid_graphs.append(g)
                valid_smiles.append(smi)

        if not valid_graphs:
            st.error("No valid SMILES provided.")
            st.stop()

        # ── MODE 1: Single model ─────────────────────────────
        if app_mode == "Single Model":
            model, _ = load_single_model(sel_model, sel_cond)
            if model is None:
                st.error(f"Checkpoint not found for {sel_model} / {COND_LABELS[sel_cond]}. "
                         "Please train first.")
                st.stop()

            probs, pic50s, preds = predict(model, valid_graphs, Y_MEAN, Y_STD, cls_threshold)
            rows = []
            for smi, prob, pic50, pred in zip(valid_smiles, probs, pic50s, preds):
                rows.append({
                    "SMILES"         : smi[:55] + ("…" if len(smi)>55 else ""),
                    "Activity"       : "🟢 Active" if pred else "🔴 Inactive",
                    "P(Active)"      : round(float(prob), 4),
                    "Predicted pIC50": round(float(pic50), 3),
                })
            df_res = pd.DataFrame(rows)
            st.dataframe(df_res, use_container_width=True, height=260)
            st.download_button("⬇️ Download CSV",
                               df_res.to_csv(index=False).encode(),
                               "predictions.csv", "text/csv")

        # ── MODE 2: Compare all 4 models (one condition) ─────
        elif app_mode == "Compare All Models":
            sel_cond = st.selectbox("Condition to compare across",
                                    COND_NAMES, format_func=lambda c: COND_LABELS[c])

            all_rows = []
            model_probs = {}

            for mname in MODEL_NAMES:
                model, _ = load_single_model(mname, sel_cond)
                if model is None:
                    st.warning(f"{mname} checkpoint missing — skipping.")
                    continue
                probs, pic50s, preds = predict(model, valid_graphs,
                                               Y_MEAN, Y_STD, cls_threshold)
                model_probs[mname] = probs
                for smi, prob, pic50, pred in zip(valid_smiles, probs, pic50s, preds):
                    all_rows.append({
                        "Model"          : mname,
                        "SMILES"         : smi[:45]+"…" if len(smi)>45 else smi,
                        "Activity"       : "🟢 Active" if pred else "🔴 Inactive",
                        "P(Active)"      : round(float(prob),4),
                        "Predicted pIC50": round(float(pic50),3),
                    })

            df_res = pd.DataFrame(all_rows)
            st.dataframe(df_res, use_container_width=True, height=300)
            st.download_button("⬇️ Download CSV",
                               df_res.to_csv(index=False).encode(),
                               "model_comparison.csv","text/csv")

            # Bar chart: P(Active) per model per molecule
            if len(valid_smiles) <= 10 and len(model_probs) > 1:
                st.markdown("**P(Active) comparison across models**")
                fig, ax = plt.subplots(figsize=(8, 3.5))
                x = np.arange(len(valid_smiles))
                w = 0.18; offsets = [-1.5,-0.5,0.5,1.5]
                for off, mname in zip(offsets, list(model_probs.keys())):
                    ax.bar(x + off*w, model_probs[mname], w,
                           label=mname, color=MODEL_COLORS[mname])
                ax.axhline(cls_threshold, color="red", ls="--", lw=1,
                           label=f"Threshold={cls_threshold}")
                ax.set_xticks(x)
                ax.set_xticklabels([s[:20]+"…" if len(s)>20 else s
                                    for s in valid_smiles], rotation=25, ha="right")
                ax.set_ylim(0,1.1); ax.set_ylabel("P(Active)")
                ax.set_title("Model Comparison — Probability of Activity",fontsize=12)
                ax.legend(fontsize=9); plt.tight_layout()
                buf = io.BytesIO(); fig.savefig(buf,format="png",dpi=120); buf.seek(0)
                st.image(buf, use_column_width=True)
                plt.close()

        # ── MODE 3: Compare all 4 conditions (one model) ─────
        elif app_mode == "Compare All Conditions":
            sel_model = st.selectbox("Model to compare across conditions",
                                     MODEL_NAMES, index=3)

            all_rows   = []
            cond_probs = {}

            for cond in COND_NAMES:
                model, _ = load_single_model(sel_model, cond)
                if model is None:
                    st.warning(f"{COND_LABELS[cond]} checkpoint missing — skipping.")
                    continue
                probs, pic50s, preds = predict(model, valid_graphs,
                                               Y_MEAN, Y_STD, cls_threshold)
                cond_probs[cond] = probs
                for smi, prob, pic50, pred in zip(valid_smiles, probs, pic50s, preds):
                    all_rows.append({
                        "Condition"      : COND_LABELS[cond],
                        "SMILES"         : smi[:45]+"…" if len(smi)>45 else smi,
                        "Activity"       : "🟢 Active" if pred else "🔴 Inactive",
                        "P(Active)"      : round(float(prob),4),
                        "Predicted pIC50": round(float(pic50),3),
                    })

            df_res = pd.DataFrame(all_rows)
            st.dataframe(df_res, use_container_width=True, height=300)
            st.download_button("⬇️ Download CSV",
                               df_res.to_csv(index=False).encode(),
                               "condition_comparison.csv","text/csv")

            # Bar chart: P(Active) per condition per molecule
            if len(valid_smiles) <= 10 and len(cond_probs) > 1:
                st.markdown("**P(Active) comparison across conditions**")
                COND_COLORS_LOCAL = {
                    "Random_NoAug":"#4C72B0","Random_Aug":"#DD8452",
                    "Scaffold_NoAug":"#55A868","Scaffold_Aug":"#C44E52",
                }
                fig, ax = plt.subplots(figsize=(8,3.5))
                x = np.arange(len(valid_smiles))
                w = 0.18; offsets = [-1.5,-0.5,0.5,1.5]
                for off, cond in zip(offsets, list(cond_probs.keys())):
                    ax.bar(x+off*w, cond_probs[cond], w,
                           label=COND_LABELS[cond], color=COND_COLORS_LOCAL[cond])
                ax.axhline(cls_threshold, color="red", ls="--", lw=1,
                           label=f"Threshold={cls_threshold}")
                ax.set_xticks(x)
                ax.set_xticklabels([s[:20]+"…" if len(s)>20 else s
                                    for s in valid_smiles], rotation=25, ha="right")
                ax.set_ylim(0,1.1); ax.set_ylabel("P(Active)")
                ax.set_title(f"{sel_model} — Condition Comparison", fontsize=12)
                ax.legend(fontsize=8); plt.tight_layout()
                buf = io.BytesIO(); fig.savefig(buf,format="png",dpi=120); buf.seek(0)
                st.image(buf, use_column_width=True)
                plt.close()

    elif predict_btn:
        st.info("Please enter at least one SMILES string.")

# ─── Results Summary Tab ─────────────────────────────────────
st.divider()
with st.expander("📈 View Training Results Summary (from results_summary.csv)", expanded=False):
    if os.path.exists("results_summary.csv"):
        df_sum = pd.read_csv("results_summary.csv")
        # Highlight best per metric per condition
        st.dataframe(df_sum, use_container_width=True)

        # Quick AUC heatmap from summary
        if "AUC" in df_sum.columns and "Model" in df_sum.columns:
            import matplotlib
            matplotlib.use("Agg")
            pivot = df_sum.pivot(index="Model", columns="Condition", values="AUC")
            fig, ax = plt.subplots(figsize=(9,3.5))
            import seaborn as sns
            sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlOrRd",
                        vmin=0.5, vmax=1.0, linewidths=0.5, ax=ax)
            ax.set_title("AUC — All Models × All Conditions", fontsize=12, fontweight="bold")
            plt.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf,format="png",dpi=120); buf.seek(0)
            st.image(buf, use_column_width=True); plt.close()
    else:
        st.info("Run training first — `results_summary.csv` will appear here.")

# ─── Footer ──────────────────────────────────────────────────
st.divider()
st.caption(
    "MSMP-GNN (GCN + GATv2 + TransformerConv)  ·  "
    f"Active threshold: pIC50 ≥ {THRESHOLD_PIC50}  ·  "
    "5-fold Cross-Validation  ·  Augmentation: SMILES enum + edge dropout + feature noise"
)
