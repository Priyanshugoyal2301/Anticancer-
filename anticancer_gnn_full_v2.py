# ============================================================
# ANTICANCER GNN — FULL RESEARCH PIPELINE v2
# 4 Models × 4 Conditions = 16 Training Runs
#
# Models    : GCNModel | GATModel | GTModel | MSMPModel
# Conditions: Random-NoAug | Random-Aug | Scaffold-NoAug | Scaffold-Aug
#
# All 44+ visualisations comparing every model × condition
# ============================================================

# ============================================================
# CELL 1 — INSTALL (uncomment in Colab / Kaggle)
# ============================================================
# !pip install torch-geometric rdkit-pypi tqdm seaborn -q
# !pip install torch-scatter torch-sparse \
#   -f https://data.pyg.org/whl/torch-2.0.0+cu118.html -q

# ============================================================
# CELL 2 — IMPORTS & CONFIG
# ============================================================
import os, random, warnings, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, QED
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GCNConv, GATv2Conv, TransformerConv,
    global_mean_pool, global_max_pool,
)

from sklearn.metrics import (
    accuracy_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix, r2_score, mean_squared_error,
    mean_absolute_error, roc_curve, precision_recall_curve,
    f1_score, precision_score, recall_score, balanced_accuracy_score,
)
from sklearn.model_selection import KFold
from sklearn.manifold import TSNE
from tqdm import tqdm

# ── Reproducibility ──────────────────────────────────────────
SEED = 42
def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(SEED)

# ── Device ────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ── Config ────────────────────────────────────────────────────
DATA_PATH      = "Anticancer_dataset (1).csv"
BATCH_SIZE     = 64
EPOCHS         = 150          # lower to 80 for quick test on CPU
LR             = 3e-4
WEIGHT_DECAY   = 1e-4
NUM_FOLDS      = 5
THRESHOLD      = 7.0          # pIC50 threshold for active/inactive
PLOT_DIR       = "plots"
MODEL_DIR      = "saved_models"
os.makedirs(PLOT_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Visual palette — one colour per MODEL (consistent across all plots)
MODEL_NAMES  = ["GCN", "GAT", "GT", "MSMP"]
MODEL_COLORS = {"GCN": "#4C72B0", "GAT": "#DD8452",
                "GT" : "#55A868", "MSMP": "#C44E52"}

COND_NAMES   = ["Random_NoAug", "Random_Aug",
                "Scaffold_NoAug", "Scaffold_Aug"]
COND_LABELS  = {
    "Random_NoAug"   : "Random | No Aug",
    "Random_Aug"     : "Random | Aug",
    "Scaffold_NoAug" : "Scaffold | No Aug",
    "Scaffold_Aug"   : "Scaffold | Aug",
}
COND_COLORS  = {
    "Random_NoAug"   : "#4C72B0",
    "Random_Aug"     : "#DD8452",
    "Scaffold_NoAug" : "#55A868",
    "Scaffold_Aug"   : "#C44E52",
}

plt.style.use("seaborn-v0_8-whitegrid")

def savefig(name):
    p = os.path.join(PLOT_DIR, f"{name}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {p}")

print("Config OK. Starting pipeline …")

# ============================================================
# CELL 3 — DATASET
# ============================================================
df = pd.read_csv(DATA_PATH)
df.rename(columns={"PIC50": "pIC50", "Smiles": "SMILES"}, inplace=True)
df = df.dropna(subset=["SMILES", "pIC50"]).reset_index(drop=True)
df["active"] = (df["pIC50"] >= THRESHOLD).astype(int)

print(f"Dataset: {df.shape[0]} molecules | "
      f"Active: {df['active'].sum()} | Inactive: {(df['active']==0).sum()}")

# ============================================================
# CELL 4 — FEATURE ENGINEERING
# ============================================================

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
        int(hyb == Chem.rdchem.HybridizationType.SP),
        int(hyb == Chem.rdchem.HybridizationType.SP2),
        int(hyb == Chem.rdchem.HybridizationType.SP3), c,
    ]

def bond_features(bond):
    bt = bond.GetBondType()
    return [
        float(bt == Chem.rdchem.BondType.SINGLE),
        float(bt == Chem.rdchem.BondType.DOUBLE),
        float(bt == Chem.rdchem.BondType.TRIPLE),
        float(bt == Chem.rdchem.BondType.AROMATIC),
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

def get_features(mol):
    try: AllChem.ComputeGasteigerCharges(mol)
    except: pass

    x     = [atom_features(a) for a in mol.GetAtoms()]
    edges, eattr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edges += [[i,j],[j,i]]; eattr += [bf,bf]

    fp   = np.array(AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024),
                    dtype=np.float32)
    desc = np.array(mol_descriptors(mol), dtype=np.float32)

    ei = np.array(edges, dtype=np.int64).T if edges else np.zeros((2,0), dtype=np.int64)
    ea = np.array(eattr, dtype=np.float32) if eattr else np.zeros((0,6), dtype=np.float32)
    return np.array(x, dtype=np.float32), ei, ea, fp, desc

# ── Build graphs ─────────────────────────────────────────────
print("Building graphs …")
graphs, failed = [], 0

for _, row in tqdm(df.iterrows(), total=len(df)):
    try:
        mol = Chem.MolFromSmiles(row["SMILES"])
        if mol is None: failed += 1; continue
        mol = Chem.AddHs(mol)
        x, ei, ea, fp, desc = get_features(mol)
        if np.isnan(x).any(): failed += 1; continue
        graphs.append(Data(
            x          = torch.tensor(x,    dtype=torch.float),
            edge_index = torch.tensor(ei,   dtype=torch.long),
            edge_attr  = torch.tensor(ea,   dtype=torch.float),
            y_cls      = torch.tensor([row["active"]], dtype=torch.float),
            y_reg      = torch.tensor([row["pIC50"]],  dtype=torch.float),
            fp         = torch.tensor(fp,   dtype=torch.float).unsqueeze(0),
            desc       = torch.tensor(desc, dtype=torch.float).unsqueeze(0),
            smiles     = row["SMILES"],
        ))
    except: failed += 1

NODE_DIM = graphs[0].x.shape[1]
EDGE_DIM = graphs[0].edge_attr.shape[1] if graphs[0].edge_attr.numel() else 6
print(f"Graphs: {len(graphs)} | Failed: {failed} | Node dim: {NODE_DIM} | Edge dim: {EDGE_DIM}")

# Global normalisation
y_all  = np.array([g.y_reg.item() for g in graphs])
Y_MEAN = float(y_all.mean())
Y_STD  = float(y_all.std()) + 1e-8
print(f"Y_MEAN={Y_MEAN:.4f}  Y_STD={Y_STD:.4f}")

# ============================================================
# CELL 5 — AUGMENTATION DATASET
# ============================================================

class ChemAugDataset(torch.utils.data.Dataset):
    def __init__(self, graphs, augment=False):
        self.graphs  = graphs
        self.augment = augment

    def __len__(self): return len(self.graphs)

    def _rand_smiles(self, smi):
        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(mol, doRandom=True) if mol else smi

    def __getitem__(self, idx):
        g = self.graphs[idx]
        if not self.augment: return g

        strat = np.random.choice(["smiles","drop_edge","noise"],
                                 p=[0.50, 0.25, 0.25])
        if strat == "smiles":
            smi = self._rand_smiles(g.smiles)
            mol = Chem.MolFromSmiles(smi)
            if mol is None: return g
            try:
                mol = Chem.AddHs(mol)
                x, ei, ea, fp, desc = get_features(mol)
                return Data(
                    x=torch.tensor(x,dtype=torch.float),
                    edge_index=torch.tensor(ei,dtype=torch.long),
                    edge_attr=torch.tensor(ea,dtype=torch.float),
                    y_cls=g.y_cls, y_reg=g.y_reg,
                    fp=torch.tensor(fp,dtype=torch.float).unsqueeze(0),
                    desc=torch.tensor(desc,dtype=torch.float).unsqueeze(0),
                    smiles=smi)
            except: return g

        elif strat == "drop_edge":
            d = g.clone()
            if d.edge_index.shape[1] > 0:
                mask = torch.rand(d.edge_index.shape[1]) > 0.15
                d.edge_index = d.edge_index[:,mask]
                d.edge_attr  = d.edge_attr[mask]
            return d
        else:
            d = g.clone()
            d.x = d.x + torch.randn_like(d.x) * 0.05
            return d

# ============================================================
# CELL 6 — SPLITS
# ============================================================

def random_splits(n, folds=NUM_FOLDS):
    kf = KFold(n_splits=folds, shuffle=True, random_state=SEED)
    return [(tr.tolist(), te.tolist()) for tr, te in kf.split(range(n))]

def scaffold_splits(smiles_list, folds=NUM_FOLDS):
    scaffolds = {}
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        sc  = (MurckoScaffold.MurckoScaffoldSmiles(
                   mol=mol, includeChirality=False)
               if mol else "")
        scaffolds.setdefault(sc, []).append(i)

    groups = list(scaffolds.values())
    random.Random(SEED).shuffle(groups)

    bins      = [[] for _ in range(folds)]
    bin_sizes = [0] * folds
    for g in groups:
        smallest = int(np.argmin(bin_sizes))
        bins[smallest].extend(g)
        bin_sizes[smallest] += len(g)

    splits = []
    for i in range(folds):
        te = bins[i]
        tr = [idx for j in range(folds) if j != i for idx in bins[j]]
        splits.append((tr, te))
    return splits

rand_splits  = random_splits(len(graphs))
scaff_splits = scaffold_splits([g.smiles for g in graphs])

SPLITS_MAP = {
    "Random_NoAug"  : rand_splits,
    "Random_Aug"    : rand_splits,
    "Scaffold_NoAug": scaff_splits,
    "Scaffold_Aug"  : scaff_splits,
}
AUG_MAP = {
    "Random_NoAug"  : False,
    "Random_Aug"    : True,
    "Scaffold_NoAug": False,
    "Scaffold_Aug"  : True,
}

# ============================================================
# CELL 7 — MODEL DEFINITIONS (4 independent models)
# ============================================================

# ── Shared regression head (deep, SiLU) ──────────────────────
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

# ── Shared classification head ────────────────────────────────
class ClsHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512,256),    nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256,1),
        )
    def forward(self, x): return self.net(x)

# ── Residual GNN block ────────────────────────────────────────
class GNNBlock(nn.Module):
    def __init__(self, conv, hidden, use_edge_attr=True):
        super().__init__()
        self.conv          = conv
        self.use_edge_attr = use_edge_attr
        self.norm          = nn.LayerNorm(hidden)
        self.drop          = nn.Dropout(0.1)
    def forward(self, x, ei, ea=None):
        h = (self.conv(x, ei, ea) if self.use_edge_attr and ea is not None
             else self.conv(x, ei))
        return self.drop(F.gelu(self.norm(h)))

# ────────────────────────────────────────────────────────────
# MODEL 1 — GCN
# ────────────────────────────────────────────────────────────
class GCNModel(nn.Module):
    def __init__(self, node_dim, edge_dim):
        super().__init__()
        H = 128
        self.conv1 = GNNBlock(GCNConv(node_dim, H), H, use_edge_attr=False)
        self.conv2 = GNNBlock(GCNConv(H, H),        H, use_edge_attr=False)
        fuse       = H*2 + 1024 + 9
        self.norm  = nn.LayerNorm(fuse)
        self.cls   = ClsHead(fuse)
        self.reg   = RegHead(fuse)

    def forward(self, data):
        x  = self.conv2(self.conv1(data.x, data.edge_index), data.edge_index)
        p  = torch.cat([global_mean_pool(x, data.batch),
                        global_max_pool(x,  data.batch)], dim=1)
        fp = data.fp.view(p.size(0),-1);  de = data.desc.view(p.size(0),-1)
        f  = self.norm(torch.cat([p,fp,de], dim=1))
        return self.cls(f).squeeze(-1), self.reg(f).squeeze(-1)

# ────────────────────────────────────────────────────────────
# MODEL 2 — GAT (GATv2)
# ────────────────────────────────────────────────────────────
class GATModel(nn.Module):
    def __init__(self, node_dim, edge_dim):
        super().__init__()
        H = 128
        self.conv1 = GNNBlock(GATv2Conv(node_dim, H, heads=4, concat=False, edge_dim=edge_dim), H)
        self.conv2 = GNNBlock(GATv2Conv(H,        H, heads=4, concat=False, edge_dim=edge_dim), H)
        fuse       = H*2 + 1024 + 9
        self.norm  = nn.LayerNorm(fuse)
        self.cls   = ClsHead(fuse)
        self.reg   = RegHead(fuse)

    def forward(self, data):
        ea = data.edge_attr
        x  = self.conv2(self.conv1(data.x, data.edge_index, ea), data.edge_index, ea)
        p  = torch.cat([global_mean_pool(x, data.batch),
                        global_max_pool(x,  data.batch)], dim=1)
        fp = data.fp.view(p.size(0),-1);  de = data.desc.view(p.size(0),-1)
        f  = self.norm(torch.cat([p,fp,de], dim=1))
        return self.cls(f).squeeze(-1), self.reg(f).squeeze(-1)

# ────────────────────────────────────────────────────────────
# MODEL 3 — GT (Graph Transformer)
# ────────────────────────────────────────────────────────────
class GTModel(nn.Module):
    def __init__(self, node_dim, edge_dim):
        super().__init__()
        H = 128
        self.conv1 = GNNBlock(TransformerConv(node_dim, H, heads=4, concat=False, edge_dim=edge_dim), H)
        self.conv2 = GNNBlock(TransformerConv(H,        H, heads=4, concat=False, edge_dim=edge_dim), H)
        fuse       = H*2 + 1024 + 9
        self.norm  = nn.LayerNorm(fuse)
        self.cls   = ClsHead(fuse)
        self.reg   = RegHead(fuse)

    def forward(self, data):
        ea = data.edge_attr
        x  = self.conv2(self.conv1(data.x, data.edge_index, ea), data.edge_index, ea)
        p  = torch.cat([global_mean_pool(x, data.batch),
                        global_max_pool(x,  data.batch)], dim=1)
        fp = data.fp.view(p.size(0),-1);  de = data.desc.view(p.size(0),-1)
        f  = self.norm(torch.cat([p,fp,de], dim=1))
        return self.cls(f).squeeze(-1), self.reg(f).squeeze(-1)

# ────────────────────────────────────────────────────────────
# MODEL 4 — MSMP (GCN + GAT + GT in parallel)
# ────────────────────────────────────────────────────────────
class MSMPModel(nn.Module):
    def __init__(self, node_dim, edge_dim):
        super().__init__()
        H = 128
        self.gcn1 = GNNBlock(GCNConv(node_dim, H), H, use_edge_attr=False)
        self.gcn2 = GNNBlock(GCNConv(H, H),        H, use_edge_attr=False)
        self.gat1 = GNNBlock(GATv2Conv(node_dim, H, heads=4, concat=False, edge_dim=edge_dim), H)
        self.gat2 = GNNBlock(GATv2Conv(H,        H, heads=4, concat=False, edge_dim=edge_dim), H)
        self.gt1  = GNNBlock(TransformerConv(node_dim, H, heads=4, concat=False, edge_dim=edge_dim), H)
        self.gt2  = GNNBlock(TransformerConv(H,        H, heads=4, concat=False, edge_dim=edge_dim), H)
        fuse      = H*3*2 + 1024 + 9
        self.norm = nn.LayerNorm(fuse)
        self.cls  = ClsHead(fuse)
        self.reg  = RegHead(fuse)

    def forward(self, data):
        x, ei, ea = data.x, data.edge_index, data.edge_attr
        b = data.batch
        g1 = self.gcn2(self.gcn1(x, ei),    ei)
        g2 = self.gat2(self.gat1(x, ei, ea), ei, ea)
        g3 = self.gt2( self.gt1( x, ei, ea), ei, ea)
        def pool(h):
            return torch.cat([global_mean_pool(h,b), global_max_pool(h,b)], dim=1)
        p  = torch.cat([pool(g1), pool(g2), pool(g3)], dim=1)
        fp = data.fp.view(p.size(0),-1);  de = data.desc.view(p.size(0),-1)
        f  = self.norm(torch.cat([p,fp,de], dim=1))
        return self.cls(f).squeeze(-1), self.reg(f).squeeze(-1)

MODEL_CLASSES = {
    "GCN" : GCNModel,
    "GAT" : GATModel,
    "GT"  : GTModel,
    "MSMP": MSMPModel,
}

def build_model(name):
    return MODEL_CLASSES[name](NODE_DIM, EDGE_DIM).to(device)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None: nn.init.zeros_(m.bias)

# ============================================================
# CELL 8 — TRAIN / EVAL UTILITIES
# ============================================================

def compute_loss(cls_out, reg_out, batch):
    yc = batch.y_cls.squeeze().to(device)
    yr = batch.y_reg.squeeze().to(device)
    yr_norm = (yr - Y_MEAN) / Y_STD
    l_cls = F.binary_cross_entropy_with_logits(cls_out, yc)
    l_reg = F.smooth_l1_loss(reg_out, yr_norm)
    return 0.5 * l_cls + 1.2 * l_reg

def train_epoch(model, loader, opt, scaler=None):
    model.train()
    total = 0.0
    for batch in loader:
        batch = batch.to(device)
        batch.fp = batch.fp.to(device); batch.desc = batch.desc.to(device)
        opt.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                cls_o, reg_o = model(batch)
                loss = compute_loss(cls_o, reg_o, batch)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        else:
            cls_o, reg_o = model(batch)
            loss = compute_loss(cls_o, reg_o, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        total += loss.item()
    return total / max(len(loader), 1)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    probs_l, labels_l, rp_l, rt_l = [], [], [], []

    for batch in loader:
        batch = batch.to(device)
        batch.fp = batch.fp.to(device); batch.desc = batch.desc.to(device)
        cls_o, reg_o = model(batch)
        probs_l.extend(torch.sigmoid(cls_o).cpu().numpy())
        labels_l.extend(batch.y_cls.cpu().numpy())
        rp_l.extend((reg_o.cpu().numpy() * Y_STD + Y_MEAN))
        rt_l.extend(batch.y_reg.cpu().numpy())

    probs  = np.array(probs_l,  dtype=np.float32)
    labels = np.array(labels_l, dtype=np.float32)
    rp     = np.array(rp_l,     dtype=np.float32)
    rt     = np.array(rt_l,     dtype=np.float32)

    if len(np.unique(labels)) < 2:
        auc   = 0.5
        fpr_v = np.array([0.,1.]); tpr_v = np.array([0.,1.])
        best_t = 0.5
    else:
        fpr_v, tpr_v, thr_v = roc_curve(labels, probs)
        best_t = float(thr_v[np.argmax(tpr_v - fpr_v)])
        auc    = float(roc_auc_score(labels, probs))

    preds = (probs >= best_t).astype(int)
    cm    = confusion_matrix(labels, preds).ravel()
    tn, fp_, fn, tp = cm if len(cm)==4 else [0,0,0,len(labels)]

    prec_v, rec_v, _ = (precision_recall_curve(labels, probs)
                        if len(np.unique(labels)) > 1
                        else ([1.,0.],[0.,1.],None))

    cls_m = {
        "ACC"        : float(accuracy_score(labels, preds)),
        "AUC"        : auc,
        "MCC"        : float(matthews_corrcoef(labels, preds)),
        "Sensitivity": float(tp/(tp+fn+1e-8)),
        "Specificity": float(tn/(tn+fp_+1e-8)),
        "F1"         : float(f1_score(labels, preds, zero_division=0)),
        "Precision"  : float(precision_score(labels, preds, zero_division=0)),
        "BAcc"       : float(balanced_accuracy_score(labels, preds)),
    }
    reg_m = {
        "RMSE": float(np.sqrt(mean_squared_error(rt, rp))),
        "MAE" : float(mean_absolute_error(rt, rp)),
        "R2"  : float(r2_score(rt, rp)),
    }
    curves = {
        "fpr": fpr_v, "tpr": tpr_v,
        "prec": np.array(prec_v, dtype=np.float32),
        "rec" : np.array(rec_v,  dtype=np.float32),
        "probs": probs, "labels": labels,
        "reg_pred": rp, "reg_true": rt,
        "cm": np.array([[tn,fp_],[fn,tp]]),
    }
    return cls_m, reg_m, curves

# ============================================================
# CELL 9 — MAIN TRAINING LOOP  (4 models × 4 conditions)
# ============================================================
# Schema:
#   all_results[model_name][cond_name] = {
#       fold_cls, fold_reg, fold_curves, loss_history
#   }

USE_AMP = torch.cuda.is_available()

all_results = {m: {} for m in MODEL_NAMES}

for model_name in MODEL_NAMES:
    print(f"\n{'='*70}")
    print(f"  MODEL: {model_name}")
    print(f"{'='*70}")

    for cond_name in COND_NAMES:
        splits = SPLITS_MAP[cond_name]
        augment = AUG_MAP[cond_name]

        print(f"\n  ── Condition: {COND_LABELS[cond_name]} ──")

        fold_cls, fold_reg, fold_curves, fold_loss = [], [], [], []
        best_state = None
        best_auc_global = -1.0

        for fold_idx, (tr_idx, te_idx) in enumerate(splits):
            print(f"    Fold {fold_idx+1}/{NUM_FOLDS}", end=" ")

            tr_data = ChemAugDataset([graphs[i] for i in tr_idx], augment=augment)
            te_data = ChemAugDataset([graphs[i] for i in te_idx], augment=False)

            tr_loader = DataLoader(tr_data, batch_size=BATCH_SIZE,
                                   shuffle=True, drop_last=True, num_workers=0)
            te_loader = DataLoader(te_data, batch_size=BATCH_SIZE,
                                   shuffle=False, num_workers=0)

            model = build_model(model_name)
            model.apply(init_weights)

            opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=LR*10, total_steps=EPOCHS, pct_start=0.2)
            scaler = torch.cuda.amp.GradScaler() if USE_AMP else None

            epoch_losses = []
            fold_best_auc = -1.0
            fold_best_state = None

            for ep in range(EPOCHS):
                loss = train_epoch(model, tr_loader, opt, scaler)
                sched.step()
                epoch_losses.append(loss)

                # Quick val check every 30 epochs to save best weights
                if (ep+1) % 30 == 0 or ep == EPOCHS-1:
                    cls_m, reg_m, _ = evaluate(model, te_loader)
                    if cls_m["AUC"] > fold_best_auc:
                        fold_best_auc   = cls_m["AUC"]
                        fold_best_state = {k: v.cpu().clone()
                                           for k, v in model.state_dict().items()}

            # Restore best weights for this fold
            model.load_state_dict({k: v.to(device)
                                   for k, v in fold_best_state.items()})
            cls_m, reg_m, curves = evaluate(model, te_loader)
            fold_cls.append(cls_m); fold_reg.append(reg_m)
            fold_curves.append(curves); fold_loss.append(epoch_losses)

            print(f"| AUC={cls_m['AUC']:.3f}  R²={reg_m['R2']:.3f}  RMSE={reg_m['RMSE']:.3f}")

            if fold_best_auc > best_auc_global:
                best_auc_global = fold_best_auc
                best_state      = fold_best_state

        # Save best model checkpoint
        torch.save({
            "state_dict": best_state, "model_name": model_name,
            "node_dim": NODE_DIM, "edge_dim": EDGE_DIM,
            "Y_MEAN": Y_MEAN, "Y_STD": Y_STD, "condition": cond_name,
        }, os.path.join(MODEL_DIR, f"{model_name}_{cond_name}_best.pt"))

        avg_cls = {k: np.mean([f[k] for f in fold_cls]) for k in fold_cls[0]}
        avg_reg = {k: np.mean([f[k] for f in fold_reg]) for k in fold_reg[0]}
        print(f"  ✅ Mean → AUC={avg_cls['AUC']:.3f}  "
              f"R²={avg_reg['R2']:.3f}  RMSE={avg_reg['RMSE']:.3f}")

        all_results[model_name][cond_name] = {
            "fold_cls"    : fold_cls,
            "fold_reg"    : fold_reg,
            "fold_curves" : fold_curves,
            "loss_history": fold_loss,
        }

print("\n\n✅ ALL 16 RUNS COMPLETE")

# Save meta
json.dump({"Y_MEAN": Y_MEAN, "Y_STD": Y_STD,
           "NODE_DIM": NODE_DIM, "EDGE_DIM": EDGE_DIM,
           "THRESHOLD": THRESHOLD},
          open(os.path.join(MODEL_DIR, "meta.json"), "w"), indent=2)

# ============================================================
# HELPER UTILITIES FOR PLOTTING
# ============================================================

def avg_m(model_name, cond, key, split="cls"):
    return np.mean([f[key] for f in all_results[model_name][cond][f"fold_{split}"]])

def agg(model_name, cond):
    curves = all_results[model_name][cond]["fold_curves"]
    return {
        "probs"   : np.concatenate([c["probs"]    for c in curves]),
        "labels"  : np.concatenate([c["labels"]   for c in curves]),
        "reg_pred": np.concatenate([c["reg_pred"] for c in curves]),
        "reg_true": np.concatenate([c["reg_true"] for c in curves]),
    }

# ============================================================
# ▶ PLOT 1 — Chemical Space Analysis
# ============================================================
print("\n━━━ Plot 1: Chemical Space ━━━")

# 1-A Tanimoto similarity
fps_rdk = []
for g in graphs[:500]:
    mol = Chem.MolFromSmiles(g.smiles)
    if mol: fps_rdk.append(Chem.RDKFingerprint(mol))
sims = [DataStructs.TanimotoSimilarity(fps_rdk[i], fps_rdk[j])
        for i in range(len(fps_rdk)) for j in range(i+1, len(fps_rdk))]

fig, ax = plt.subplots(figsize=(8,4))
ax.hist(sims, bins=50, color="#4C72B0", edgecolor="white", alpha=0.85)
ax.axvline(np.mean(sims), color="#C44E52", lw=2,
           label=f"Mean = {np.mean(sims):.3f}")
ax.set_xlabel("Tanimoto Similarity",fontsize=12)
ax.set_ylabel("Count",fontsize=12)
ax.set_title("Pairwise Tanimoto Similarity Distribution",fontsize=14,fontweight="bold")
ax.legend(); plt.tight_layout(); savefig("01A_tanimoto")

# 1-B pIC50 distribution
fig, ax = plt.subplots(figsize=(8,4))
ax.hist(df["pIC50"].values, bins=50, color="#55A868", edgecolor="white", alpha=0.85)
ax.axvline(THRESHOLD, color="#C44E52", lw=2, ls="--",
           label=f"Threshold = {THRESHOLD}")
ax.set_xlabel("pIC50",fontsize=12); ax.set_ylabel("Count",fontsize=12)
ax.set_title("pIC50 Distribution",fontsize=14,fontweight="bold")
ax.legend(); plt.tight_layout(); savefig("01B_pic50_dist")

# ============================================================
# ▶ PLOT 2 — Classification Metrics Bar Charts
#    One chart per metric, all 4 models × 4 conditions grouped
# ============================================================
print("\n━━━ Plot 2: Classification Metric Bars ━━━")

CLS_METRICS = ["ACC","AUC","MCC","Sensitivity","Specificity","F1","Precision","BAcc"]
x      = np.arange(len(COND_NAMES))
width  = 0.18
offsets = [-1.5, -0.5, 0.5, 1.5]

for metric in CLS_METRICS:
    fig, ax = plt.subplots(figsize=(11,5))
    for moff, mname in zip(offsets, MODEL_NAMES):
        vals = [avg_m(mname, c, metric, "cls") for c in COND_NAMES]
        bars = ax.bar(x + moff*width, vals, width,
                      label=mname, color=MODEL_COLORS[mname],
                      edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABELS[c] for c in COND_NAMES], rotation=10)
    ax.set_ylim(0, 1.15); ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f"Classification — {metric}  (5-fold mean)",
                 fontsize=13, fontweight="bold")
    ax.legend(title="Model", fontsize=9); plt.tight_layout()
    savefig(f"02_{metric}_bar")

# ============================================================
# ▶ PLOT 3 — ROC Curves
#    Individual (4 models on 1 plot per condition) + combined
# ============================================================
print("\n━━━ Plot 3: ROC Curves ━━━")

# Per condition: all 4 models on one figure
for cond in COND_NAMES:
    fig, ax = plt.subplots(figsize=(6,5))
    for mname in MODEL_NAMES:
        d = agg(mname, cond)
        if len(np.unique(d["labels"])) < 2: continue
        fpr, tpr, _ = roc_curve(d["labels"], d["probs"])
        auc_v = roc_auc_score(d["labels"], d["probs"])
        ax.plot(fpr, tpr, lw=2, color=MODEL_COLORS[mname],
                label=f"{mname} (AUC={auc_v:.3f})")
    ax.plot([0,1],[0,1],"k--",lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"ROC — {COND_LABELS[cond]}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9); plt.tight_layout()
    savefig(f"03_ROC_{cond}")

# Combined: 4 subplots (one per condition), all models
fig, axes = plt.subplots(2,2,figsize=(12,10))
for ax, cond in zip(axes.flat, COND_NAMES):
    for mname in MODEL_NAMES:
        d = agg(mname, cond)
        if len(np.unique(d["labels"])) < 2: continue
        fpr, tpr, _ = roc_curve(d["labels"], d["probs"])
        auc_v = roc_auc_score(d["labels"], d["probs"])
        ax.plot(fpr, tpr, lw=2, color=MODEL_COLORS[mname],
                label=f"{mname} ({auc_v:.3f})")
    ax.plot([0,1],[0,1],"k--",lw=1)
    ax.set_title(COND_LABELS[cond], fontsize=11, fontweight="bold")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(fontsize=8)
fig.suptitle("ROC Curves — All Models × All Conditions",
             fontsize=14, fontweight="bold")
plt.tight_layout(); savefig("03_ROC_combined_all")

# ============================================================
# ▶ PLOT 4 — Regression Scatter (Predicted vs True)
# ============================================================
print("\n━━━ Plot 4: Regression Scatter ━━━")

# Per condition: 4 subplots (one per model)
for cond in COND_NAMES:
    fig, axes = plt.subplots(1,4,figsize=(18,5))
    for ax, mname in zip(axes, MODEL_NAMES):
        d  = agg(mname, cond)
        rp, rt = d["reg_pred"], d["reg_true"]
        r2 = r2_score(rt, rp)
        rmse = np.sqrt(mean_squared_error(rt, rp))
        ax.scatter(rt, rp, alpha=0.35, s=8,
                   color=MODEL_COLORS[mname], rasterized=True)
        mn,mx = min(rt.min(),rp.min()), max(rt.max(),rp.max())
        ax.plot([mn,mx],[mn,mx],"r--",lw=1.5)
        ax.set_title(f"{mname}\nR²={r2:.3f}  RMSE={rmse:.3f}",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("True pIC50"); ax.set_ylabel("Pred pIC50")
    fig.suptitle(f"Predicted vs True — {COND_LABELS[cond]}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(); savefig(f"04_RegScatter_{cond}")

# ============================================================
# ▶ PLOT 5 — Confusion Matrices
# ============================================================
print("\n━━━ Plot 5: Confusion Matrices ━━━")

for cond in COND_NAMES:
    fig, axes = plt.subplots(1,4,figsize=(18,4))
    for ax, mname in zip(axes, MODEL_NAMES):
        d     = agg(mname, cond)
        preds = (d["probs"] >= 0.5).astype(int)
        cm    = confusion_matrix(d["labels"], preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Inactive","Active"],
                    yticklabels=["Inactive","Active"], ax=ax)
        ax.set_title(mname, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.suptitle(f"Confusion Matrices — {COND_LABELS[cond]}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(); savefig(f"05_ConfMatrix_{cond}")

# ============================================================
# ▶ PLOT 6 — Fold-Level Violin Plots (ACC & AUC)
# ============================================================
print("\n━━━ Plot 6: Fold Violin Plots ━━━")

for metric in ["ACC","AUC"]:
    fig, axes = plt.subplots(1,4,figsize=(18,5))
    for ax, cond in zip(axes, COND_NAMES):
        data_v = {mname: [f[metric]
                           for f in all_results[mname][cond]["fold_cls"]]
                  for mname in MODEL_NAMES}
        df_v = pd.DataFrame(data_v)
        sns.violinplot(data=df_v,
                       palette=[MODEL_COLORS[m] for m in MODEL_NAMES],
                       ax=ax, inner="point")
        ax.set_title(COND_LABELS[cond], fontsize=10, fontweight="bold")
        ax.set_ylabel(metric)
    fig.suptitle(f"{metric} Distribution Across Folds — All Models",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(); savefig(f"06_Violin_{metric}")

# ============================================================
# ▶ PLOT 7 — Augmentation Impact per Model
# ============================================================
print("\n━━━ Plot 7: Augmentation Impact ━━━")

AUG_METRICS = ["ACC","AUC","MCC","F1"]

for split_type in ["Random","Scaffold"]:
    no_aug = f"{split_type}_NoAug"
    aug    = f"{split_type}_Aug"
    fig, ax = plt.subplots(figsize=(10,5))
    x_pos = np.arange(len(AUG_METRICS))
    w = 0.18
    for moff, mname in zip(offsets, MODEL_NAMES):
        gains = [avg_m(mname,aug,m,"cls") - avg_m(mname,no_aug,m,"cls")
                 for m in AUG_METRICS]
        ax.bar(x_pos + moff*w, gains, w,
               label=mname, color=MODEL_COLORS[mname], edgecolor="white")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x_pos); ax.set_xticklabels(AUG_METRICS)
    ax.set_ylabel("Gain (Aug − NoAug)"); ax.set_xlabel("Metric")
    ax.set_title(f"Augmentation Effect — {split_type} Split  (per model)",
                 fontsize=13, fontweight="bold")
    ax.legend(title="Model"); plt.tight_layout()
    savefig(f"07_AugEffect_{split_type}")

# ============================================================
# ▶ PLOT 8 — Random vs Scaffold Comparison per Model
# ============================================================
print("\n━━━ Plot 8: Random vs Scaffold ━━━")

COMP_METRICS = ["ACC","AUC","MCC","F1","Sensitivity","Specificity"]

for aug_tag, rand_c, scaff_c in [
    ("NoAug","Random_NoAug","Scaffold_NoAug"),
    ("Aug",  "Random_Aug",  "Scaffold_Aug"),
]:
    fig, axes = plt.subplots(1,4,figsize=(20,5))
    for ax, mname in zip(axes, MODEL_NAMES):
        x_pos = np.arange(len(COMP_METRICS))
        vr = [avg_m(mname, rand_c,  m, "cls") for m in COMP_METRICS]
        vs = [avg_m(mname, scaff_c, m, "cls") for m in COMP_METRICS]
        ax.bar(x_pos-0.2, vr, 0.4, label="Random",   color=COND_COLORS[rand_c])
        ax.bar(x_pos+0.2, vs, 0.4, label="Scaffold",  color=COND_COLORS[scaff_c])
        ax.set_xticks(x_pos); ax.set_xticklabels(COMP_METRICS, rotation=30)
        ax.set_ylim(0,1.15); ax.set_title(mname, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
    fig.suptitle(f"Random vs Scaffold — {aug_tag}  (all models)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(); savefig(f"08_RandVsScaff_{aug_tag}")

# ============================================================
# ▶ PLOT 9 — R² Heatmap  (models × conditions × folds)
# ============================================================
print("\n━━━ Plot 9: R² Heatmap ━━━")

# Mean R² per model × condition
r2_mat = np.array([[avg_m(m, c, "R2", "reg")
                    for c in COND_NAMES] for m in MODEL_NAMES])
fig, ax = plt.subplots(figsize=(10,4))
sns.heatmap(r2_mat, annot=True, fmt=".3f", cmap="RdYlGn",
            xticklabels=[COND_LABELS[c] for c in COND_NAMES],
            yticklabels=MODEL_NAMES, vmin=-0.1, vmax=1.0,
            linewidths=0.5, ax=ax)
ax.set_title("R² Heatmap — All Models × All Conditions (5-fold mean)",
             fontsize=13, fontweight="bold")
plt.tight_layout(); savefig("09_R2_heatmap")

# Per-model fold-level R² heatmap
for mname in MODEL_NAMES:
    r2_folds = np.array([[f["R2"] for f in all_results[mname][c]["fold_reg"]]
                          for c in COND_NAMES])
    fig, ax = plt.subplots(figsize=(9,4))
    sns.heatmap(r2_folds, annot=True, fmt=".3f", cmap="RdYlGn",
                xticklabels=[f"Fold {i+1}" for i in range(NUM_FOLDS)],
                yticklabels=[COND_LABELS[c] for c in COND_NAMES],
                vmin=-0.1, vmax=1.0, linewidths=0.5, ax=ax)
    ax.set_title(f"R² per Fold — {mname}",fontsize=12,fontweight="bold")
    plt.tight_layout(); savefig(f"09_R2_heatmap_{mname}")

# ============================================================
# ▶ PLOT 10 — Chemical Similarity vs Fold Accuracy
# ============================================================
print("\n━━━ Plot 10: Sim vs Accuracy ━━━")

fig, ax = plt.subplots(figsize=(7,5))
for mname in MODEL_NAMES:
    sim_means, accs = [], []
    for fold_idx, (tr_idx, te_idx) in enumerate(rand_splits):
        tr_fps, te_fps = [], []
        for i in tr_idx[:80]:
            mol = Chem.MolFromSmiles(graphs[i].smiles)
            if mol: tr_fps.append(Chem.RDKFingerprint(mol))
        for i in te_idx[:20]:
            mol = Chem.MolFromSmiles(graphs[i].smiles)
            if mol: te_fps.append(Chem.RDKFingerprint(mol))
        cross = [DataStructs.TanimotoSimilarity(tf,trf)
                 for tf in te_fps for trf in tr_fps]
        sim_means.append(np.mean(cross) if cross else 0)
        accs.append(all_results[mname]["Random_NoAug"]["fold_cls"][fold_idx]["ACC"])
    ax.scatter(sim_means, accs, label=mname, color=MODEL_COLORS[mname], s=60, zorder=5)
ax.set_xlabel("Mean Tanimoto (Test vs Train)",fontsize=11)
ax.set_ylabel("Fold Accuracy",fontsize=11)
ax.set_title("Chemical Similarity vs Fold Accuracy",fontsize=13,fontweight="bold")
ax.legend(title="Model"); plt.tight_layout(); savefig("10_SimVsAccuracy")

# ============================================================
# ▶ PLOT 11 — Training Dynamics (Loss Convergence)
# ============================================================
print("\n━━━ Plot 11: Training Convergence ━━━")

for cond in COND_NAMES:
    fig, axes = plt.subplots(1,4,figsize=(18,4))
    for ax, mname in zip(axes, MODEL_NAMES):
        for fi, lh in enumerate(all_results[mname][cond]["loss_history"]):
            ax.plot(lh, alpha=0.6, label=f"Fold {fi+1}")
        ax.set_title(mname, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(fontsize=7)
    fig.suptitle(f"Loss Convergence — {COND_LABELS[cond]}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(); savefig(f"11_Convergence_{cond}")

# ============================================================
# ▶ PLOT 12 — Residual Analysis
# ============================================================
print("\n━━━ Plot 12: Residual Analysis ━━━")

for cond in COND_NAMES:
    # 12-A: scatter
    fig, axes = plt.subplots(1,4,figsize=(18,5))
    for ax, mname in zip(axes, MODEL_NAMES):
        d = agg(mname, cond)
        resid = d["reg_pred"] - d["reg_true"]
        ax.scatter(d["reg_true"], resid, alpha=0.3, s=8,
                   color=MODEL_COLORS[mname], rasterized=True)
        ax.axhline(0, color="red", lw=1.5, ls="--")
        ax.set_xlabel("True pIC50"); ax.set_ylabel("Residual")
        ax.set_title(mname, fontsize=11, fontweight="bold")
    fig.suptitle(f"Residuals vs True pIC50 — {COND_LABELS[cond]}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(); savefig(f"12A_ResidScatter_{cond}")

    # 12-B: histogram
    fig, axes = plt.subplots(1,4,figsize=(18,4))
    for ax, mname in zip(axes, MODEL_NAMES):
        d = agg(mname, cond)
        resid = d["reg_pred"] - d["reg_true"]
        ax.hist(resid, bins=40, color=MODEL_COLORS[mname],
                edgecolor="white", alpha=0.8)
        ax.axvline(0, color="red", lw=1.5, ls="--")
        ax.set_xlabel("Residual"); ax.set_ylabel("Count")
        ax.set_title(mname, fontsize=11, fontweight="bold")
    fig.suptitle(f"Residual Distribution — {COND_LABELS[cond]}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(); savefig(f"12B_ResidHist_{cond}")

# ============================================================
# ▶ PLOT 13 — t-SNE Embeddings (fingerprint space)
# ============================================================
print("\n━━━ Plot 13: t-SNE ━━━")

N_TSNE  = min(1500, len(graphs))
fp_mat  = np.vstack([g.fp.squeeze().numpy() for g in graphs[:N_TSNE]])
pic50s  = np.array([g.y_reg.item() for g in graphs[:N_TSNE]])
acts    = np.array([g.y_cls.item() for g in graphs[:N_TSNE]])

print("  Running t-SNE …")
tsne_emb = TSNE(n_components=2, perplexity=30, random_state=SEED,
                max_iter=1000).fit_transform(fp_mat)

fig, ax = plt.subplots(figsize=(7,5))
sc = ax.scatter(tsne_emb[:,0], tsne_emb[:,1], c=pic50s,
                cmap="viridis", s=10, alpha=0.7, rasterized=True)
plt.colorbar(sc, ax=ax, label="pIC50")
ax.set_title("t-SNE coloured by pIC50",fontsize=13,fontweight="bold")
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
plt.tight_layout(); savefig("13A_tSNE_pIC50")

fig, ax = plt.subplots(figsize=(7,5))
colors_act = ["#4C72B0" if a==0 else "#DD8452" for a in acts]
ax.scatter(tsne_emb[:,0], tsne_emb[:,1], c=colors_act,
           s=10, alpha=0.7, rasterized=True)
ax.legend(handles=[mpatches.Patch(color="#4C72B0",label="Inactive"),
                   mpatches.Patch(color="#DD8452",label="Active")])
ax.set_title("t-SNE coloured by Activity",fontsize=13,fontweight="bold")
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
plt.tight_layout(); savefig("13B_tSNE_activity")

# ============================================================
# ▶ PLOT 14 — Precision-Recall Curves
# ============================================================
print("\n━━━ Plot 14: PR Curves ━━━")

for cond in COND_NAMES:
    fig, ax = plt.subplots(figsize=(6,5))
    for mname in MODEL_NAMES:
        d = agg(mname, cond)
        if len(np.unique(d["labels"])) < 2: continue
        prec, rec, _ = precision_recall_curve(d["labels"], d["probs"])
        ap = float(np.trapz(prec[::-1], rec[::-1]))
        ax.plot(rec, prec, lw=2, color=MODEL_COLORS[mname],
                label=f"{mname} (AP={ap:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"PR Curve — {COND_LABELS[cond]}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9); plt.tight_layout()
    savefig(f"14_PR_{cond}")

# Combined: 4 subplots
fig, axes = plt.subplots(2,2,figsize=(12,10))
for ax, cond in zip(axes.flat, COND_NAMES):
    for mname in MODEL_NAMES:
        d = agg(mname, cond)
        if len(np.unique(d["labels"])) < 2: continue
        prec, rec, _ = precision_recall_curve(d["labels"], d["probs"])
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
# ▶ PLOT 15 — Augmentation Gain Summary (Random vs Scaffold)
# ============================================================
print("\n━━━ Plot 15: Augmentation Gain Summary ━━━")

GM = ["ACC","AUC","MCC","F1"]
x_pos = np.arange(len(GM)); w = 0.18

fig, axes = plt.subplots(1,2,figsize=(14,5))
for ax, (split_type, no_aug_c, aug_c) in zip(axes, [
    ("Random",   "Random_NoAug",   "Random_Aug"),
    ("Scaffold", "Scaffold_NoAug", "Scaffold_Aug"),
]):
    for moff, mname in zip(offsets, MODEL_NAMES):
        gains = [avg_m(mname,aug_c,m,"cls") - avg_m(mname,no_aug_c,m,"cls")
                 for m in GM]
        ax.bar(x_pos + moff*w, gains, w,
               label=mname, color=MODEL_COLORS[mname], edgecolor="white")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x_pos); ax.set_xticklabels(GM)
    ax.set_ylabel("Gain (Aug − NoAug)"); ax.set_xlabel("Metric")
    ax.set_title(f"{split_type} Split — Aug Gain per Model",
                 fontsize=12, fontweight="bold")
    ax.legend(title="Model", fontsize=9)
plt.tight_layout(); savefig("15_AugGain_summary")

# ============================================================
# ▶ BONUS — Model Ranking Heatmap (AUC across all 16 runs)
# ============================================================
print("\n━━━ Bonus: Model Ranking Heatmap ━━━")

auc_mat = np.array([[avg_m(m,c,"AUC","cls") for c in COND_NAMES]
                    for m in MODEL_NAMES])
fig, ax = plt.subplots(figsize=(10,4))
sns.heatmap(auc_mat, annot=True, fmt=".3f", cmap="YlOrRd",
            xticklabels=[COND_LABELS[c] for c in COND_NAMES],
            yticklabels=MODEL_NAMES, vmin=0.5, vmax=1.0,
            linewidths=0.5, ax=ax)
ax.set_title("AUC Heatmap — All 4 Models × All 4 Conditions",
             fontsize=13, fontweight="bold")
plt.tight_layout(); savefig("00_AUC_ranking_heatmap")

rmse_mat = np.array([[avg_m(m,c,"RMSE","reg") for c in COND_NAMES]
                     for m in MODEL_NAMES])
fig, ax = plt.subplots(figsize=(10,4))
sns.heatmap(rmse_mat, annot=True, fmt=".3f", cmap="RdYlGn_r",
            xticklabels=[COND_LABELS[c] for c in COND_NAMES],
            yticklabels=MODEL_NAMES, linewidths=0.5, ax=ax)
ax.set_title("RMSE Heatmap — All 4 Models × All 4 Conditions",
             fontsize=13, fontweight="bold")
plt.tight_layout(); savefig("00_RMSE_ranking_heatmap")

# ============================================================
# CELL 10 — FINAL RESULTS TABLE
# ============================================================
print("\n\n" + "="*70)
print("  FINAL RESULTS SUMMARY (5-fold mean)")
print("="*70)

rows = []
for mname in MODEL_NAMES:
    for cond in COND_NAMES:
        r = {"Model": mname, "Condition": COND_LABELS[cond]}
        for k in ["ACC","AUC","MCC","F1","Sensitivity","Specificity","BAcc"]:
            r[k] = round(avg_m(mname, cond, k, "cls"), 4)
        for k in ["RMSE","MAE","R2"]:
            r[k] = round(avg_m(mname, cond, k, "reg"), 4)
        rows.append(r)

summary = pd.DataFrame(rows)
pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print(summary.to_string(index=False))
summary.to_csv("results_summary.csv", index=False)
print("\n✅ Saved → results_summary.csv")
print(f"✅ Plots  → {PLOT_DIR}/")
print(f"✅ Models → {MODEL_DIR}/")
print("\n→ Deploy: streamlit run app.py")
