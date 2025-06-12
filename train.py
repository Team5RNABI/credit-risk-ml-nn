#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train.py

Script completo optimizado para:
  - Supresión de warnings de Optuna Distributions
  - Carga y limpieza de datos
  - Entrenamiento Challenger (Optuna + PyTorch) en modo eager
  - Optimización de F₂‐score (impagos)
  - Métricas por época: Loss, ROC‐AUC, PR‐AUC, Brier, Accuracy, Precision, Recall, F1, F2
  - Monitorización de GPU
  - Evaluación final en test set
  - Guardado de artefactos en reports/: best_model.pth, optuna_study.pkl, metrics.json, history.json
"""

import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="optuna.distributions"
)

import json, joblib, torch, torch.multiprocessing as mp
import numpy as np, pandas as pd
from pathlib import Path
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score
)

# ── 1. Configuración global ──────────────────────────────────────────
SEED         = 42
BATCH        = 2048
EPOCHS_FINAL = 15
PATIENCE     = 3
BETA_F       = 2.0
CUDA         = torch.cuda.is_available()
DEVICE       = torch.device("cuda" if CUDA else "cpu")
REPORT_DIR   = Path("reports"); REPORT_DIR.mkdir(exist_ok=True)
torch.manual_seed(SEED); np.random.seed(SEED)

def show_gpu(stage=""):
    if CUDA:
        used  = torch.cuda.memory_allocated()/1024**2
        total = torch.cuda.get_device_properties(0).total_memory/1024**2
        print(f"{stage} → VRAM used: {used:.0f}/{total:.0f} MiB")
    else:
        print(f"{stage} → GPU not available")

# ── 2. Dataset con embeddings ────────────────────────────────────────
class CreditDataset(Dataset):
    def __init__(self, df, num_cols, cat_cols, num_stats=None, cat_maps=None):
        self.y = torch.tensor(df["target"].values, dtype=torch.float32)
        if num_stats is None:
            self.means = df[num_cols].mean()
            self.stds  = df[num_cols].std().replace(0,1)
        else:
            self.means, self.stds = num_stats
        self.X_num = torch.tensor(
            ((df[num_cols] - self.means)/self.stds).values, dtype=torch.float32
        )
        if cat_maps is None:
            self.cat_maps = {
                c: {cat:i for i,cat in enumerate(df[c].astype("category").cat.categories)}
                for c in cat_cols
            }
        else:
            self.cat_maps = cat_maps
        self.X_cat = torch.tensor(
            np.stack([df[c].map(self.cat_maps[c]).fillna(-1).astype(int).values
                      for c in cat_cols], axis=1),
            dtype=torch.long
        )
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return (self.X_num[idx], self.X_cat[idx]), self.y[idx]

# ── 3. Modelo RiskNN ─────────────────────────────────────────────────
class RiskNN(nn.Module):
    def __init__(self, num_features, cat_dims, emb_dims, hidden=(256,128), dropout=0.3):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(d,e) for d,e in zip(cat_dims, emb_dims)])
        in_dim = num_features + sum(emb_dims)
        layers = []
        for h in hidden:
            layers += [nn.Linear(in_dim,h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim,1))
        self.net = nn.Sequential(*layers)
    def forward(self, x_num, x_cat):
        emb = [m(x_cat[:,i]) for i,m in enumerate(self.emb)]
        x   = torch.cat(emb + [x_num], dim=1)
        return self.net(x).squeeze(1)

# ── 4. Funciones de entrenamiento y evaluación ────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler, scheduler=None):
    model.train(); total_loss=0.0
    for (x_num,x_cat), y in loader:
        x_num, x_cat, y = x_num.to(DEVICE), x_cat.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        with autocast():
            logits = model(x_num, x_cat)
            loss   = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        if scheduler: scheduler.step()
        total_loss += loss.item()*y.size(0)
    return total_loss/len(loader.dataset)

@torch.no_grad()
def evaluate_probs(model, loader):
    model.eval(); ys, ps = [], []
    for (x_num,x_cat), y in loader:
        x_num, x_cat = x_num.to(DEVICE), x_cat.to(DEVICE)
        logits = model(x_num, x_cat)
        ys.append(y.cpu().numpy()); ps.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)

def evaluate_metrics(y_true, y_prob):
    roc   = roc_auc_score(y_true, y_prob)
    pr    = average_precision_score(y_true, y_prob)
    brier= brier_score_loss(y_true, y_prob)
    y_pred=(y_prob>=0.5).astype(int)
    acc   = accuracy_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    f2    = fbeta_score(y_true, y_pred, beta=BETA_F, zero_division=0)
    return {"roc":roc,"pr":pr,"brier":brier,
            "accuracy":acc,"precision":prec,
            "recall":rec,"f1":f1,"f2":f2}

# ── 5. main() ────────────────────────────────────────────────────────
def main():
    mp.set_start_method('spawn', force=True)

    # 5.1 Carga y limpieza data
    df = pd.read_csv("data/processed/data_loan_complete.csv")
    df = df[df["loan_status_bin"].notna()].copy()
    df["target"] = df["loan_status_bin"].astype(int)
    df.drop(columns=["loan_status_bin"], inplace=True)

    # 5.2 Splits
    train_df,test_df = train_test_split(df, test_size=0.15,
                                        stratify=df["target"], random_state=SEED)
    train_df,valid_df= train_test_split(train_df, test_size=0.1765,
                                        stratify=train_df["target"], random_state=SEED)

    # 5.3 Columnas y dims
    categorical_cols = [c for c in df.columns if df[c].dtype=="object"]
    numerical_cols   = [c for c in df.columns if c not in categorical_cols+["target"]]
    cat_dims = [train_df[c].nunique() for c in categorical_cols]
    emb_dims = [min(50,d//2+1) for d in cat_dims]

    # 5.4 Datasets
    train_ds = CreditDataset(train_df, numerical_cols, categorical_cols)
    valid_ds = CreditDataset(valid_df, numerical_cols, categorical_cols,
                             num_stats=(train_ds.means,train_ds.stds),
                             cat_maps=train_ds.cat_maps)
    test_ds  = CreditDataset(test_df,  numerical_cols, categorical_cols,
                             num_stats=(train_ds.means,train_ds.stds),
                             cat_maps=train_ds.cat_maps)

    # 5.5 DataLoaders
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=0, pin_memory=False)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH, shuffle=False,
                              num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False,
                              num_workers=0, pin_memory=False)

    # 5.6 History
    history = {"train_loss":[],"valid_loss":[]}
    for k in ["roc","pr","brier","accuracy","precision","recall","f1","f2"]:
        history[k]=[]

    # 5.7 Función objetivo enclosure
    def objective(trial):
        hidden  = trial.suggest_categorical("hidden", ((256,128),(128,64),(256,128,64)))
        dropout = trial.suggest_float("dropout",0.1,0.5)
        lr      = trial.suggest_float("lr",1e-4,5e-3,log=True)
        wd      = trial.suggest_float("wd",1e-6,1e-3,log=True)

        c0,c1 = np.bincount(train_df["target"].astype(int))
        pos_w = torch.tensor([c0/c1],device=DEVICE)

        model     = RiskNN(len(numerical_cols),cat_dims,emb_dims,hidden,dropout).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        optimizer = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd)
        scaler    = GradScaler()
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=lr,
                                                        total_steps=len(train_loader)*5)

        best_f2=0.0
        for epoch in range(5):
            loss = train_one_epoch(model,train_loader,criterion,optimizer,scaler,scheduler)
            yv,pv = evaluate_probs(model,valid_loader)
            m = evaluate_metrics(yv,pv)
            print(
                f"Trial {trial.number} Ep {epoch+1}/5 → "
                f"Loss={loss:.4f} | "
                f"ROC={m['roc']:.4f} | PR={m['pr']:.4f} | Brier={m['brier']:.4f} | "
                f"Acc={m['accuracy']:.4f} | Prec={m['precision']:.4f} | "
                f"Rec={m['recall']:.4f} | F1={m['f1']:.4f} | F2={m['f2']:.4f}"
            )

            trial.report(m["f2"],epoch)
            if trial.should_prune(): raise optuna.TrialPruned()
            best_f2 = max(best_f2,m["f2"])
        return best_f2

    # 5.8 Optimización HPO
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED),
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=1))
    study.optimize(objective, n_trials=20)

    # 5.9 Entrenamiento final con histórico
    best = study.best_trial.params
    model = RiskNN(len(numerical_cols),cat_dims,emb_dims,best["hidden"],best["dropout"]).to(DEVICE)
    c0,c1 = np.bincount(train_df["target"].astype(int))
    pos_w = torch.tensor([c0/c1],device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = torch.optim.AdamW(model.parameters(),lr=best["lr"],weight_decay=best["wd"])
    scaler    = GradScaler()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=best["lr"],
                                                    total_steps=len(train_loader)*EPOCHS_FINAL)

    for epoch in range(EPOCHS_FINAL):
        tr_loss = train_one_epoch(model,train_loader,criterion,optimizer,scaler,scheduler)
        history["train_loss"].append(tr_loss)

        yv,pv    = evaluate_probs(model,valid_loader)
        val_loss = brier_score_loss(yv,pv)
        history["valid_loss"].append(val_loss)

        m=evaluate_metrics(yv,pv)
        for k in history:
            if k not in ["train_loss","valid_loss"]:
                history[k].append(m[k])

        print(
            f"Epoch {epoch+1} → "
            f"tr_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | "
            f"ROC={m['roc']:.4f} | PR={m['pr']:.4f} | Brier={m['brier']:.4f} | "
            f"Acc={m['accuracy']:.4f} | Prec={m['precision']:.4f} | "
            f"Rec={m['recall']:.4f} | F1={m['f1']:.4f} | F2={m['f2']:.4f}"
        )


    # guardar histórico
    with open(REPORT_DIR/"history.json","w") as f:
        json.dump(history,f,indent=2)

    # 5.10 Evaluación final y guardado
    model.load_state_dict(torch.load(REPORT_DIR/"best_model.pth"))
    yt,pt = evaluate_probs(model,test_loader)
    mt = evaluate_metrics(yt,pt)
    with open(REPORT_DIR/"metrics.json","w") as f:
        json.dump(mt,f,indent=2)
    joblib.dump(study, REPORT_DIR/"optuna_study.pkl")

    print("\n✅ Artefactos guardados en reports/")

if __name__=="__main__":
    main()
