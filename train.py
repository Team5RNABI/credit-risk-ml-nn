#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py

Script completo optimizado para:
  - Suprimir warnings de Optuna Distributions
  - Carga, limpieza y partición de datos
  - Definición de dataset PyTorch con embeddings para categóricas
  - Arquitectura de red neuronal con embeddings + capas densas + Dropout
  - Búsqueda automática de hiperparámetros (Optuna) optimizando F2-score
  - Entrenamiento final con OneCycleLR y early-stopping sobre F2
  - Monitorización de VRAM GPU en cada fase
  - Evaluación y guardado de artefactos: best_model.pth, history.json,
    optuna_study.pkl, metrics.json
"""

import warnings
# Ignoramos avisos específicos de Optuna que no afectan al entrenamiento
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="optuna.distributions"
)

import json            # Para guardar diccionarios de métricas e histórico
import joblib          # Para serializar el estudio de Optuna
import torch           # Framework de Deep Learning
import torch.multiprocessing as mp  # Para DataLoader en Windows
import numpy as np     # Álgebra lineal y cálculos vectorizados
import pandas as pd    # Manejo de tablas de datos
from pathlib import Path

# Módulos de PyTorch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler  # Mixed precision

# Optuna para HPO
import optuna

# Métricas de sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    accuracy_score, precision_score, recall_score,
    f1_score, fbeta_score
)

# ── 1. Configuración global ──────────────────────────────────────────
SEED         = 42    # Semilla única para reproducibilidad total
BATCH        = 2048  # Tamaño de lote de datos para DataLoader
EPOCHS_FINAL = 15    # Nº de épocas en el entrenamiento definitivo
PATIENCE     = 3     # Paciencia para early-stopping (F2)
BETA_F       = 2.0   # Beta para fbeta_score (prioriza recall)

# Detectar y seleccionar GPU si está disponible
CUDA   = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")

# Carpeta donde guardamos todos los artefactos
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

# Fijar semilla para numpy y PyTorch (GPU + CPU)
torch.manual_seed(SEED)
np.random.seed(SEED)

# Función de ayuda para mostrar uso de memoria GPU en MiB
def show_gpu(stage=""):
    if CUDA:
        used  = torch.cuda.memory_allocated() / 1024**2
        total = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"[{stage}] VRAM used: {used:.0f}/{total:.0f} MiB")
    else:
        print(f"[{stage}] GPU not available")


# ── 2. Dataset con embeddings para variables categóricas ─────────────
class CreditDataset(Dataset):
    """
    PyTorch Dataset que:
      1) Normaliza variables numéricas (media 0, desvío 1)
      2) Convierte variables categóricas a enteros según mapa
      3) Devuelve tensores X_num, X_cat, y para entrenamiento/eval
    """
    def __init__(self, df, num_cols, cat_cols,
                 num_stats=None, cat_maps=None):
        # Vector de etiquetas (float32) 0/1
        self.y = torch.tensor(df["target"].values,
                              dtype=torch.float32)

        # --- Escalado para numéricas ---
        if num_stats is None:
            # Calculamos media y desvío sobre train
            self.means = df[num_cols].mean()
            self.stds  = df[num_cols].std().replace(0,1)
        else:
            # Reutilizamos estadísticos del train
            self.means, self.stds = num_stats

        # Aplicamos (x - mean) / std y convertimos a tensor
        self.X_num = torch.tensor(
            ((df[num_cols] - self.means) / self.stds).values,
            dtype=torch.float32
        )

        # --- Mapeo de categorías a índices ---
        if cat_maps is None:
            # Creamos diccionario para cada columna categórica
            self.cat_maps = {
                c: {cat: i for i, cat in enumerate(
                            df[c].astype('category')
                                 .cat.categories)}
                for c in cat_cols
            }
        else:
            self.cat_maps = cat_maps

        # Convertimos a índices enteros y a tensor long
        self.X_cat = torch.tensor(
            np.stack([
                df[c]
                  .map(self.cat_maps[c])  # Reemplazamos string→int
                  .fillna(-1)             # Missing categories
                  .astype(int)
                  .values
                for c in cat_cols
            ], axis=1),
            dtype=torch.long
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Devuelve ((X_num, X_cat), y) para el índice idx
        return (self.X_num[idx], self.X_cat[idx]), self.y[idx]


# ── 3. Definición de la red neuronal (RiskNN) ─────────────────────────
class RiskNN(nn.Module):
    """
    Red neuronal que combina:
      - Embeddings para variables categóricas
      - Capas densas con BatchNorm, GELU y Dropout
    """
    def __init__(self,
                 num_features,  # nº de variables numéricas
                 cat_dims,      # lista: nº categorías por col
                 emb_dims,      # lista: dimensión embedding por col
                 hidden=(256,128),  # neuronas por capa densa
                 dropout=0.3       # prob. dropout
                ):
        super().__init__()
        # Creamos un nn.Embedding por cada categórica
        self.emb = nn.ModuleList([
            nn.Embedding(dim, emb)
            for dim, emb in zip(cat_dims, emb_dims)
        ])

        # Dimensión de entrada = suma de embeddings + num_features
        in_dim = num_features + sum(emb_dims)

        # Construimos la pila de capas densas
        layers = []
        for h in hidden:
            layers += [
                nn.Linear(in_dim, h),       # capa lineal
                nn.BatchNorm1d(h),          # normaliza activaciones
                nn.GELU(),                  # función de activación
                nn.Dropout(dropout)         # regularización
            ]
            in_dim = h  # la salida de esta capa es la entrada de la siguiente

        # Capa final lineal a 1 neurona (logit)
        layers.append(nn.Linear(in_dim, 1))

        # Secuenciamos todo en self.net
        self.net = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        # 1) Obtenemos embeddings por columna categórica
        embs = [m(x_cat[:, i]) for i, m in enumerate(self.emb)]
        # 2) Concatenamos embeddings + datos numéricos
        x = torch.cat(embs + [x_num], dim=1)
        # 3) Pasamos por la red y aplanamos a vector
        return self.net(x).squeeze(1)


# ── 4. Función de entrenamiento y evaluación ─────────────────────────

def train_one_epoch(model, loader, criterion, optimizer,
                    scaler, scheduler=None):
    """
    Entrena 1 época completa:
      - Mixed precision (autocast + GradScaler)
      - scheduler.step() opcional (OneCycleLR)
      - Devuelve loss promedio sobre el epoch
    """
    model.train()
    total_loss = 0.0

    for (x_num, x_cat), y in loader:
        # 1) Mover datos a GPU/CPU
        x_num, x_cat, y = x_num.to(DEVICE), x_cat.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        # 2) Forward + cálculo de pérdida en autocast
        with autocast():
            logits = model(x_num, x_cat)
            loss   = criterion(logits, y)

        # 3) Backward + step del optimizador escalado
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 4) Si hay scheduler (OneCycle), avanzamos 1 paso
        if scheduler:
            scheduler.step()

        total_loss += loss.item() * y.size(0)

    # 5) Promediamos sobre todos los samples
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_probs(model, loader):
    """
    Evalúa el modelo en un DataLoader de valid/test y devuelve:
      - y_true: array numpy de etiquetas
      - y_prob: array numpy de probabilidades (sigmoid)
    """
    model.eval()
    ys, ps = [], []
    for (x_num, x_cat), y in loader:
        x_num, x_cat = x_num.to(DEVICE), x_cat.to(DEVICE)
        logits = model(x_num, x_cat)
        ys.append(y.cpu().numpy())
        ps.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def evaluate_metrics(y_true, y_prob):
    """
    Dadas etiquetas y probabilidades, calcula:
      - ROC-AUC
      - PR-AUC (Precision-Recall AUC)
      - Brier score (calibración)
      - Accuracy, Precision, Recall
      - F1 y F2 (F-beta, con beta=2 para enfatizar recall)
    """
    roc    = roc_auc_score(y_true, y_prob)
    pr     = average_precision_score(y_true, y_prob)
    brier  = brier_score_loss(y_true, y_prob)

    # Umbral fijo 0.5 para clasificación binaria
    y_pred = (y_prob >= 0.5).astype(int)
    acc    = accuracy_score(y_true, y_pred)
    prec   = precision_score(y_true, y_pred, zero_division=0)
    rec    = recall_score(y_true, y_pred, zero_division=0)
    f1     = f1_score(y_true, y_pred, zero_division=0)
    f2     = fbeta_score(y_true, y_pred,
                          beta=BETA_F, zero_division=0)

    return {
        "roc": roc,
        "pr": pr,
        "brier": brier,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "f2": f2
    }


# ── 5. Función principal: carga, HPO, entrenamiento final ───────────
def main():
    # En Windows, hay que usar spawn para multiprocessing en DataLoader
    mp.set_start_method('spawn', force=True)

    # Monitoreo uso VRAM antes de cargar datos
    show_gpu("start")

    # 5.1) Carga y limpieza del CSV procesado
    df = pd.read_csv("data/processed/data_loan_complete.csv")
    # Filtrar registros sin etiqueta
    df = df[df["loan_status_bin"].notna()].copy()
    # 0 → pago, 1 → impago
    df["target"] = df["loan_status_bin"].astype(int)
    # Eliminamos la columna original
    df.drop(columns=["loan_status_bin"], inplace=True)

    # 5.2) División estratificada: train 70%, valid 15%, test 15%
    train_df, test_df  = train_test_split(
        df, test_size=0.15,
        stratify=df["target"],
        random_state=SEED
    )
    train_df, valid_df = train_test_split(
        train_df, test_size=0.1765,
        stratify=train_df["target"],
        random_state=SEED
    )
    show_gpu("after split")

    # 5.3) Identificar columnas numéricas vs categóricas
    categorical_cols = [c for c in df.columns
                        if df[c].dtype == "object"]
    numerical_cols   = [c for c in df.columns
                        if c not in categorical_cols + ["target"]]

    # Dims para embeddings: min(50, ncat//2+1)
    cat_dims = [train_df[c].nunique() for c in categorical_cols]
    emb_dims = [min(50, d//2 + 1) for d in cat_dims]

    # 5.4) Crear Datasets y DataLoaders (num_workers=0 para evitar errores)
    train_ds = CreditDataset(
        train_df, numerical_cols, categorical_cols
    )
    valid_ds = CreditDataset(
        valid_df, numerical_cols, categorical_cols,
        num_stats=(train_ds.means, train_ds.stds),
        cat_maps=train_ds.cat_maps
    )
    test_ds  = CreditDataset(
        test_df, numerical_cols, categorical_cols,
        num_stats=(train_ds.means, train_ds.stds),
        cat_maps=train_ds.cat_maps
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH,
        shuffle=True, num_workers=0, pin_memory=False
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=BATCH,
        shuffle=False, num_workers=0, pin_memory=False
    )
    test_loader  = DataLoader(
        test_ds,  batch_size=BATCH,
        shuffle=False, num_workers=0, pin_memory=False
    )

    # ── 5.5) Hyperparameter Optimization (Optuna) ────────────────
    def objective(trial):
        # a) Definir espacio de búsqueda
        hidden  = trial.suggest_categorical(
            "hidden", ((256,128),(128,64),(256,128,64))
        )
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        lr      = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        wd      = trial.suggest_float("wd", 1e-6, 1e-3, log=True)

        # b) Calcular pos_weight para BCE (balance clases)
        c0, c1 = np.bincount(train_df["target"].astype(int))
        pos_w = torch.tensor([c0/c1], device=DEVICE)

        # c) Instanciar modelo + losses + optimizadores
        model     = RiskNN(
            len(numerical_cols), cat_dims, emb_dims,
            hidden, dropout
        ).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=wd
        )
        scaler    = GradScaler()
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr,
            total_steps=len(train_loader)*5
        )

        # d) Loop de 5 épocas (rápido) — reportar F2 y pruning
        best_f2 = 0.0
        for epoch in range(5):
            loss = train_one_epoch(
                model, train_loader, criterion,
                optimizer, scaler, scheduler
            )
            yv, pv = evaluate_probs(model, valid_loader)
            m = evaluate_metrics(yv, pv)

            # Mostrar métricas en cada época
            print(
                f"Trial {trial.number} Ep {epoch+1}/5 → "
                f"Loss={loss:.4f} | ROC={m['roc']:.4f} | "
                f"PR={m['pr']:.4f} | Brier={m['brier']:.4f} | "
                f"Acc={m['accuracy']:.4f} | Prec={m['precision']:.4f} | "
                f"Rec={m['recall']:.4f} | F1={m['f1']:.4f} | F2={m['f2']:.4f}"
            )

            trial.report(m['f2'], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
            best_f2 = max(best_f2, m['f2'])

        return best_f2

    # Creamos y ejecutamos el study
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1)
    )
    study.optimize(objective, n_trials=20)

    # Guardamos el estudio completo de Optuna
    joblib.dump(study, REPORT_DIR/"optuna_study.pkl")

    # ── 5.6) Entrenamiento final con mejores HP ────────────────
    best = study.best_trial.params

    # Reconstruimos modelo, criterios y optimizadores con mejores valores
    c0, c1 = np.bincount(train_df["target"].astype(int))
    pos_w = torch.tensor([c0/c1], device=DEVICE)
    model     = RiskNN(
        len(numerical_cols), cat_dims, emb_dims,
        best["hidden"], best["dropout"]
    ).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=best["lr"], weight_decay=best["wd"]
    )
    scaler    = GradScaler()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=best["lr"],
        total_steps=len(train_loader)*EPOCHS_FINAL
    )

    # Preparamos diccionario para guardar historial de métricas
    history = {"train_loss": [], "valid_loss": []}
    for key in ["roc","pr","brier","accuracy","precision","recall","f1","f2"]:
        history[key] = []

    best_f2, counter = 0.0, 0
    show_gpu("before final training")

    # Loop de EPOCHS_FINAL épocas con early-stopping
    for epoch in range(1, EPOCHS_FINAL+1):
        tr_loss = train_one_epoch(
            model, train_loader, criterion,
            optimizer, scaler, scheduler
        )
        scheduler.step()

        yv, pv = evaluate_probs(model, valid_loader)
        m = evaluate_metrics(yv, pv)

        # Calculamos valid_loss en GPU (BCE logits directo)
        with torch.no_grad():
            logits_all, ys_all = [], []
            for (xn, xc), y in valid_loader:
                xn, xc = xn.to(DEVICE), xc.to(DEVICE)
                logits_all.append(model(xn, xc))
                ys_all.append(y.to(DEVICE))
            all_logits = torch.cat(logits_all)
            all_ys     = torch.cat(ys_all)
            val_loss   = criterion(all_logits, all_ys).item()

        # Guardar en history
        history["train_loss"].append(tr_loss)
        history["valid_loss"].append(val_loss)
        for k, v in m.items():
            history[k].append(v)

        # Imprimir resumen de la época
        print(
            f"Ep{epoch}/{EPOCHS_FINAL} → tr_loss={tr_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            + " | ".join([f"{k}={v:.4f}" for k, v in m.items()])
        )

        # Early-stopping en función de F2
        if m['f2'] > best_f2:
            best_f2, counter = m['f2'], 0
            torch.save(model.state_dict(), REPORT_DIR/"best_model.pth")
        else:
            counter += 1
            if counter >= PATIENCE:
                print("⏹️ Early stopping: no mejora en F2")
                break

    show_gpu("after final training")

    # Guardamos historial de métricas por época
    with open(REPORT_DIR/"history.json", "w") as fp:
        json.dump(history, fp, indent=2)

    # ── 5.7) Evaluación final en test set ────────────────────────
    model.load_state_dict(torch.load(REPORT_DIR/"best_model.pth"))
    yt, pt = evaluate_probs(model, test_loader)
    final_metrics = evaluate_metrics(yt, pt)

    # Guardamos métricas finales en JSON
    with open(REPORT_DIR/"metrics.json", "w") as fp:
        json.dump(final_metrics, fp, indent=2)

    print("\n✅ Entrenamiento completado. Artefactos en", REPORT_DIR)


if __name__ == "__main__":
    main()
    # Ejecutamos la función principal para iniciar el proceso de entrenamiento y evaluación