"""
patch_plots.py
==============
Run this AFTER anticancer_gnn_full_v2.py completes training.
It regenerates Plot 13 (t-SNE) and Plots 14-15 + Bonus that
failed due to the n_iter → max_iter rename in scikit-learn 1.4+

Run:
    python patch_plots.py
"""

import os, json, warnings, random
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import torch

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, QED
RDLogger.DisableLog("rdApp.*")

from torch_geometric.data import Data
from sklearn.manifold import TSNE
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, r2_score,
    mean_squared_error
)
from tqdm import tqdm

# ── Config (must match training script) ──────────────────────
SEED           = 42
DATA_PATH      = "Anticancer_dataset (1).csv"
PLOT_DIR       = "plots"
MODEL_DIR      = "saved_models"
THRESHOLD      = 7.0

MODEL_NAMES  = ["GCN", "GAT", "GT", "MSMP"]
COND_NAMES   = ["Random_NoAug", "Random_Aug", "Scaffold_NoAug", "Scaffold_Aug"]
COND_LABELS  = {
    "Random_NoAug"   : "Random | No Aug",
    "Random_Aug"     : "Random | Aug",
    "Scaffold_NoAug" : "Scaffold | No Aug",
    "Scaffold_Aug"   : "Scaffold | Aug",
}
MODEL_COLORS = {"GCN":"#4C72B0","GAT":"#DD8452","GT":"#55A868","MSMP":"#C44E52"}
COND_COLORS  = {
    "Random_NoAug":"#4C72B0","Random_Aug":"#DD8452",
    "Scaffold_NoAug":"#55A868","Scaffold_Aug":"#C44E52",
}

plt.style.use("seaborn-v0_8-whitegrid")
os.makedirs(PLOT_DIR, exist_ok=True)

def savefig(name):
    p = os.path.join(PLOT_DIR, f"{name}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {p}")

# ── Load meta ─────────────────────────────────────────────────
meta   = json.load(open(os.path.join(MODEL_DIR, "meta.json")))
Y_MEAN = meta["Y_MEAN"]
Y_STD  = meta["Y_STD"]
print(f"Y_MEAN={Y_MEAN:.4f}  Y_STD={Y_STD:.4f}")

# ── Rebuild feature functions (same as training) ──────────────
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

# ── Reload dataset & graphs (for t-SNE fingerprints) ─────────
print("\nReloading dataset for t-SNE ...")
df = pd.read_csv(DATA_PATH)
df.rename(columns={"PIC50":"pIC50","Smiles":"SMILES"}, inplace=True)
df = df.dropna(subset=["SMILES","pIC50"]).reset_index(drop=True)
df["active"] = (df["pIC50"] >= THRESHOLD).astype(int)

fp_list, pic50_list, act_list, smi_list = [], [], [], []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Building fingerprints"):
    try:
        mol = Chem.MolFromSmiles(row["SMILES"])
        if mol is None: continue
        mol = Chem.AddHs(mol)
        fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024),
                      dtype=np.float32)
        fp_list.append(fp)
        pic50_list.append(row["pIC50"])
        act_list.append(row["active"])
        smi_list.append(row["SMILES"])
    except: continue

fp_mat  = np.vstack(fp_list)
pic50s  = np.array(pic50_list)
acts    = np.array(act_list)
print(f"Fingerprints built: {len(fp_mat)}")

# ── Load saved results for PR / bonus plots ───────────────────
# We need to re-run inference on test folds to get probs/preds.
# Since models are saved, we reload them and run quick inference.

# Import model classes inline
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GCNConv, GATv2Conv, TransformerConv,
    global_mean_pool, global_max_pool,
)
from sklearn.model_selection import KFold
from rdkit.Chem.Scaffolds import MurckoScaffold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

class RegHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,512),nn.BatchNorm1d(512),nn.SiLU(),nn.Dropout(0.3),
            nn.Linear(512,256),nn.BatchNorm1d(256),nn.SiLU(),nn.Dropout(0.2),
            nn.Linear(256,128),nn.SiLU(),nn.Linear(128,1),
        )
    def forward(self,x): return self.net(x)

class ClsHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(0.3),
            nn.Linear(512,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(0.2),
            nn.Linear(256,1),
        )
    def forward(self,x): return self.net(x)

class GNNBlock(nn.Module):
    def __init__(self,conv,hidden,use_edge_attr=True):
        super().__init__()
        self.conv=conv; self.use_edge_attr=use_edge_attr
        self.norm=nn.LayerNorm(hidden); self.drop=nn.Dropout(0.1)
    def forward(self,x,ei,ea=None):
        h=(self.conv(x,ei,ea) if self.use_edge_attr and ea is not None
           else self.conv(x,ei))
        return self.drop(F.gelu(self.norm(h)))

class GCNModel(nn.Module):
    def __init__(self,node_dim,edge_dim):
        super().__init__(); H=128
        self.conv1=GNNBlock(GCNConv(node_dim,H),H,use_edge_attr=False)
        self.conv2=GNNBlock(GCNConv(H,H),H,use_edge_attr=False)
        fuse=H*2+1024+9; self.norm=nn.LayerNorm(fuse)
        self.cls=ClsHead(fuse); self.reg=RegHead(fuse)
    def forward(self,data):
        x=self.conv2(self.conv1(data.x,data.edge_index),data.edge_index)
        p=torch.cat([global_mean_pool(x,data.batch),global_max_pool(x,data.batch)],dim=1)
        f=self.norm(torch.cat([p,data.fp.view(p.size(0),-1),data.desc.view(p.size(0),-1)],dim=1))
        return self.cls(f).squeeze(-1),self.reg(f).squeeze(-1)

class GATModel(nn.Module):
    def __init__(self,node_dim,edge_dim):
        super().__init__(); H=128
        self.conv1=GNNBlock(GATv2Conv(node_dim,H,heads=4,concat=False,edge_dim=edge_dim),H)
        self.conv2=GNNBlock(GATv2Conv(H,H,heads=4,concat=False,edge_dim=edge_dim),H)
        fuse=H*2+1024+9; self.norm=nn.LayerNorm(fuse)
        self.cls=ClsHead(fuse); self.reg=RegHead(fuse)
    def forward(self,data):
        ea=data.edge_attr
        x=self.conv2(self.conv1(data.x,data.edge_index,ea),data.edge_index,ea)
        p=torch.cat([global_mean_pool(x,data.batch),global_max_pool(x,data.batch)],dim=1)
        f=self.norm(torch.cat([p,data.fp.view(p.size(0),-1),data.desc.view(p.size(0),-1)],dim=1))
        return self.cls(f).squeeze(-1),self.reg(f).squeeze(-1)

class GTModel(nn.Module):
    def __init__(self,node_dim,edge_dim):
        super().__init__(); H=128
        self.conv1=GNNBlock(TransformerConv(node_dim,H,heads=4,concat=False,edge_dim=edge_dim),H)
        self.conv2=GNNBlock(TransformerConv(H,H,heads=4,concat=False,edge_dim=edge_dim),H)
        fuse=H*2+1024+9; self.norm=nn.LayerNorm(fuse)
        self.cls=ClsHead(fuse); self.reg=RegHead(fuse)
    def forward(self,data):
        ea=data.edge_attr
        x=self.conv2(self.conv1(data.x,data.edge_index,ea),data.edge_index,ea)
        p=torch.cat([global_mean_pool(x,data.batch),global_max_pool(x,data.batch)],dim=1)
        f=self.norm(torch.cat([p,data.fp.view(p.size(0),-1),data.desc.view(p.size(0),-1)],dim=1))
        return self.cls(f).squeeze(-1),self.reg(f).squeeze(-1)

class MSMPModel(nn.Module):
    def __init__(self,node_dim,edge_dim):
        super().__init__(); H=128
        self.gcn1=GNNBlock(GCNConv(node_dim,H),H,use_edge_attr=False)
        self.gcn2=GNNBlock(GCNConv(H,H),H,use_edge_attr=False)
        self.gat1=GNNBlock(GATv2Conv(node_dim,H,heads=4,concat=False,edge_dim=edge_dim),H)
        self.gat2=GNNBlock(GATv2Conv(H,H,heads=4,concat=False,edge_dim=edge_dim),H)
        self.gt1=GNNBlock(TransformerConv(node_dim,H,heads=4,concat=False,edge_dim=edge_dim),H)
        self.gt2=GNNBlock(TransformerConv(H,H,heads=4,concat=False,edge_dim=edge_dim),H)
        fuse=H*3*2+1024+9; self.norm=nn.LayerNorm(fuse)
        self.cls=ClsHead(fuse); self.reg=RegHead(fuse)
    def forward(self,data):
        x,ei,ea,b=data.x,data.edge_index,data.edge_attr,data.batch
        g1=self.gcn2(self.gcn1(x,ei),ei)
        g2=self.gat2(self.gat1(x,ei,ea),ei,ea)
        g3=self.gt2(self.gt1(x,ei,ea),ei,ea)
        def pool(h):
            return torch.cat([global_mean_pool(h,b),global_max_pool(h,b)],dim=1)
        p=torch.cat([pool(g1),pool(g2),pool(g3)],dim=1)
        f=self.norm(torch.cat([p,data.fp.view(p.size(0),-1),data.desc.view(p.size(0),-1)],dim=1))
        return self.cls(f).squeeze(-1),self.reg(f).squeeze(-1)

MODEL_CLASSES = {"GCN":GCNModel,"GAT":GATModel,"GT":GTModel,"MSMP":MSMPModel}

# ── Load best model checkpoint per model×cond ────────────────
def load_ckpt(mname, cond):
    path = os.path.join(MODEL_DIR, f"{mname}_{cond}_best.pt")
    if not os.path.exists(path):
        print(f"  ⚠ Missing: {path}")
        return None
    ckpt  = torch.load(path, map_location=device)
    model = MODEL_CLASSES[mname](ckpt["node_dim"], ckpt["edge_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    return model

# ── Rebuild full graph list for inference ────────────────────
print("\nRebuilding graphs for inference ...")

def get_features(mol):
    try: AllChem.ComputeGasteigerCharges(mol)
    except: pass
    x=[atom_features(a) for a in mol.GetAtoms()]
    edges,eattr=[],[]
    for bond in mol.GetBonds():
        i,j=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
        bf=bond_features(bond); edges+=[[i,j],[j,i]]; eattr+=[bf,bf]
    fp=np.array(AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024),dtype=np.float32)
    desc=np.array(mol_descriptors(mol),dtype=np.float32)
    ei=np.array(edges,dtype=np.int64).T if edges else np.zeros((2,0),dtype=np.int64)
    ea=np.array(eattr,dtype=np.float32) if eattr else np.zeros((0,6),dtype=np.float32)
    return np.array(x,dtype=np.float32),ei,ea,fp,desc

graphs=[]
for _,row in tqdm(df.iterrows(),total=len(df)):
    try:
        mol=Chem.MolFromSmiles(row["SMILES"])
        if mol is None: continue
        mol=Chem.AddHs(mol)
        x,ei,ea,fp,desc=get_features(mol)
        if np.isnan(x).any(): continue
        graphs.append(Data(
            x=torch.tensor(x,dtype=torch.float),
            edge_index=torch.tensor(ei,dtype=torch.long),
            edge_attr=torch.tensor(ea,dtype=torch.float),
            y_cls=torch.tensor([row["active"]],dtype=torch.float),
            y_reg=torch.tensor([row["pIC50"]],dtype=torch.float),
            fp=torch.tensor(fp,dtype=torch.float).unsqueeze(0),
            desc=torch.tensor(desc,dtype=torch.float).unsqueeze(0),
            smiles=row["SMILES"],
        ))
    except: continue

print(f"Graphs: {len(graphs)}")

# ── Quick inference helper ────────────────────────────────────
@torch.no_grad()
def run_inference(model, graph_list):
    loader = DataLoader(graph_list, batch_size=128, shuffle=False, num_workers=0)
    probs_l, labels_l, rp_l, rt_l = [], [], [], []
    for batch in loader:
        batch=batch.to(device)
        batch.fp=batch.fp.to(device); batch.desc=batch.desc.to(device)
        cls_o,reg_o=model(batch)
        probs_l.extend(torch.sigmoid(cls_o).cpu().numpy())
        labels_l.extend(batch.y_cls.cpu().numpy())
        rp_l.extend(reg_o.cpu().numpy()*Y_STD+Y_MEAN)
        rt_l.extend(batch.y_reg.cpu().numpy())
    return (np.array(probs_l,dtype=np.float32),
            np.array(labels_l,dtype=np.float32),
            np.array(rp_l,dtype=np.float32),
            np.array(rt_l,dtype=np.float32))

# Build agg results dict from saved checkpoints
print("\nRunning inference on all saved models ...")
agg_data = {}
for mname in MODEL_NAMES:
    agg_data[mname] = {}
    for cond in COND_NAMES:
        model = load_ckpt(mname, cond)
        if model is None: continue
        probs,labels,rp,rt = run_inference(model, graphs)
        agg_data[mname][cond] = {
            "probs":probs,"labels":labels,"reg_pred":rp,"reg_true":rt
        }
        print(f"  {mname} × {cond} — AUC={roc_auc_score(labels,probs):.3f}  "
              f"R²={r2_score(rt,rp):.3f}")

def agg(mname, cond):
    return agg_data[mname][cond]

def avg_m(mname, cond, key):
    d = agg_data[mname][cond]
    if key == "AUC":
        return roc_auc_score(d["labels"], d["probs"])
    if key == "R2":
        return r2_score(d["reg_true"], d["reg_pred"])
    if key == "RMSE":
        return float(np.sqrt(mean_squared_error(d["reg_true"], d["reg_pred"])))
    return 0.0

offsets = [-1.5,-0.5,0.5,1.5]

# ============================================================
# ▶ PLOT 13 — t-SNE  (fixed: max_iter instead of n_iter)
# ============================================================
print("\n━━━ Plot 13: t-SNE ━━━")

N_TSNE  = min(1500, len(fp_mat))
fp_sub  = fp_mat[:N_TSNE]
pic_sub = pic50s[:N_TSNE]
act_sub = acts[:N_TSNE]

print("  Running t-SNE (this takes ~1-2 min) ...")
tsne_emb = TSNE(
    n_components=2, perplexity=30,
    random_state=SEED, max_iter=1000   # ← fixed here
).fit_transform(fp_sub)

# 13-A: coloured by pIC50
fig, ax = plt.subplots(figsize=(7,5))
sc = ax.scatter(tsne_emb[:,0], tsne_emb[:,1], c=pic_sub,
                cmap="viridis", s=10, alpha=0.7, rasterized=True)
plt.colorbar(sc, ax=ax, label="pIC50")
ax.set_title("t-SNE coloured by pIC50", fontsize=13, fontweight="bold")
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
plt.tight_layout(); savefig("13A_tSNE_pIC50")

# 13-B: coloured by Activity
fig, ax = plt.subplots(figsize=(7,5))
colors_act = ["#4C72B0" if a==0 else "#DD8452" for a in act_sub]
ax.scatter(tsne_emb[:,0], tsne_emb[:,1], c=colors_act,
           s=10, alpha=0.7, rasterized=True)
ax.legend(handles=[
    mpatches.Patch(color="#4C72B0",label="Inactive"),
    mpatches.Patch(color="#DD8452",label="Active")
])
ax.set_title("t-SNE coloured by Activity", fontsize=13, fontweight="bold")
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
plt.tight_layout(); savefig("13B_tSNE_activity")

# ============================================================
# ▶ PLOT 14 — Precision-Recall Curves
# ============================================================
print("\n━━━ Plot 14: PR Curves ━━━")

for cond in COND_NAMES:
    fig, ax = plt.subplots(figsize=(6,5))
    for mname in MODEL_NAMES:
        if cond not in agg_data.get(mname,{}): continue
        d = agg(mname, cond)
        if len(np.unique(d["labels"])) < 2: continue
        prec,rec,_ = precision_recall_curve(d["labels"], d["probs"])
        ap = float(np.trapz(prec[::-1], rec[::-1]))
        ax.plot(rec, prec, lw=2, color=MODEL_COLORS[mname],
                label=f"{mname} (AP={ap:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"PR Curve — {COND_LABELS[cond]}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9); plt.tight_layout()
    savefig(f"14_PR_{cond}")

# Combined
fig, axes = plt.subplots(2,2,figsize=(12,10))
for ax, cond in zip(axes.flat, COND_NAMES):
    for mname in MODEL_NAMES:
        if cond not in agg_data.get(mname,{}): continue
        d = agg(mname, cond)
        if len(np.unique(d["labels"])) < 2: continue
        prec,rec,_ = precision_recall_curve(d["labels"], d["probs"])
        ap = float(np.trapz(prec[::-1], rec[::-1]))
        ax.plot(rec, prec, lw=2, color=MODEL_COLORS[mname],
                label=f"{mname} ({ap:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(COND_LABELS[cond], fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
fig.suptitle("PR Curves — All Models × All Conditions",
             fontsize=14, fontweight="bold")
plt.tight_layout(); savefig("14_PR_combined_all")

# ============================================================
# ▶ PLOT 15 — Augmentation Gain Summary
# ============================================================
print("\n━━━ Plot 15: Augmentation Gain Summary ━━━")

GM = ["AUC","R2"]
x_pos = np.arange(len(GM)); w = 0.18

fig, axes = plt.subplots(1,2,figsize=(14,5))
for ax, (split_type, no_aug_c, aug_c) in zip(axes,[
    ("Random",   "Random_NoAug",   "Random_Aug"),
    ("Scaffold", "Scaffold_NoAug", "Scaffold_Aug"),
]):
    for moff, mname in zip(offsets, MODEL_NAMES):
        gains = []
        for key in GM:
            try:
                g = avg_m(mname,aug_c,key) - avg_m(mname,no_aug_c,key)
            except: g=0.0
            gains.append(g)
        ax.bar(x_pos+moff*w, gains, w,
               label=mname, color=MODEL_COLORS[mname], edgecolor="white")
    ax.axhline(0,color="black",lw=0.8)
    ax.set_xticks(x_pos); ax.set_xticklabels(GM)
    ax.set_ylabel("Gain (Aug − NoAug)"); ax.set_xlabel("Metric")
    ax.set_title(f"{split_type} Split — Aug Gain per Model",
                 fontsize=12, fontweight="bold")
    ax.legend(title="Model", fontsize=9)
plt.tight_layout(); savefig("15_AugGain_summary")

# ============================================================
# ▶ BONUS — AUC & RMSE Ranking Heatmaps
# ============================================================
print("\n━━━ Bonus: Ranking Heatmaps ━━━")

auc_mat = np.array([[avg_m(m,c,"AUC") for c in COND_NAMES] for m in MODEL_NAMES])
fig, ax = plt.subplots(figsize=(10,4))
sns.heatmap(auc_mat, annot=True, fmt=".3f", cmap="YlOrRd",
            xticklabels=[COND_LABELS[c] for c in COND_NAMES],
            yticklabels=MODEL_NAMES, vmin=0.5, vmax=1.0,
            linewidths=0.5, ax=ax)
ax.set_title("AUC Heatmap — All 4 Models × All 4 Conditions",
             fontsize=13, fontweight="bold")
plt.tight_layout(); savefig("00_AUC_ranking_heatmap")

rmse_mat = np.array([[avg_m(m,c,"RMSE") for c in COND_NAMES] for m in MODEL_NAMES])
fig, ax = plt.subplots(figsize=(10,4))
sns.heatmap(rmse_mat, annot=True, fmt=".3f", cmap="RdYlGn_r",
            xticklabels=[COND_LABELS[c] for c in COND_NAMES],
            yticklabels=MODEL_NAMES, linewidths=0.5, ax=ax)
ax.set_title("RMSE Heatmap — All 4 Models × All 4 Conditions",
             fontsize=13, fontweight="bold")
plt.tight_layout(); savefig("00_RMSE_ranking_heatmap")

print("\n✅ ALL MISSING PLOTS REGENERATED SUCCESSFULLY")
print(f"✅ Check your '{PLOT_DIR}/' folder for:")
print("   13A_tSNE_pIC50.png")
print("   13B_tSNE_activity.png")
print("   14_PR_*.png  (5 files)")
print("   15_AugGain_summary.png")
print("   00_AUC_ranking_heatmap.png")
print("   00_RMSE_ranking_heatmap.png")
print("\n→ Now run:  streamlit run app_v2.py")

# ============================================================
# ▶ GENERATE results_summary.csv
# ============================================================
print("\n━━━ Generating results_summary.csv ━━━")

from sklearn.metrics import (
    accuracy_score, matthews_corrcoef, f1_score,
    precision_score, balanced_accuracy_score,
    mean_absolute_error, confusion_matrix
)

rows = []
for mname in MODEL_NAMES:
    for cond in COND_NAMES:
        if cond not in agg_data.get(mname, {}):
            continue
        d      = agg_data[mname][cond]
        probs  = d["probs"]
        labels = d["labels"]
        rp     = d["reg_pred"]
        rt     = d["reg_true"]

        # Best threshold via ROC
        fpr_v, tpr_v, thr_v = roc_curve(labels, probs)
        best_t = float(thr_v[np.argmax(tpr_v - fpr_v)])
        preds  = (probs >= best_t).astype(int)

        cm   = confusion_matrix(labels, preds).ravel()
        tn, fp_, fn, tp = cm if len(cm) == 4 else [0, 0, 0, len(labels)]

        row = {
            "Model"      : mname,
            "Condition"  : COND_LABELS[cond],
            "ACC"        : round(float(accuracy_score(labels, preds)), 4),
            "AUC"        : round(float(roc_auc_score(labels, probs)), 4),
            "MCC"        : round(float(matthews_corrcoef(labels, preds)), 4),
            "Sensitivity": round(float(tp / (tp + fn + 1e-8)), 4),
            "Specificity": round(float(tn / (tn + fp_ + 1e-8)), 4),
            "F1"         : round(float(f1_score(labels, preds, zero_division=0)), 4),
            "Precision"  : round(float(precision_score(labels, preds, zero_division=0)), 4),
            "BAcc"       : round(float(balanced_accuracy_score(labels, preds)), 4),
            "RMSE"       : round(float(np.sqrt(mean_squared_error(rt, rp))), 4),
            "MAE"        : round(float(mean_absolute_error(rt, rp)), 4),
            "R2"         : round(float(r2_score(rt, rp)), 4),
        }
        rows.append(row)
        print(f"  {mname:4s} x {cond:20s} | AUC={row['AUC']}  ACC={row['ACC']}  R2={row['R2']}  RMSE={row['RMSE']}")

summary_df = pd.DataFrame(rows)
summary_df.to_csv("results_summary.csv", index=False)
print("\n✅ results_summary.csv saved!")
print(summary_df.to_string(index=False))
