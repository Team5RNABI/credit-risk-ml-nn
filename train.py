#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SISTEMA DE EVALUACIÓN DE RIESGO CREDITICIO CON DEEP LEARNING
===============================================================

Este script implementa un pipeline completo de Machine Learning para predicción 
de riesgo crediticio utilizando redes neuronales con embeddings categóricos.

ARQUITECTURA DEL SISTEMA:
-----------------------------
- Preprocesamiento avanzado con embeddings para variables categóricas
- Red neuronal híbrida (embeddings + capas densas) con regularización
- Optimización automática de hiperparámetros usando Optuna + TPE sampler
- Entrenamiento con técnicas SOTA: OneCycleLR, Mixed Precision, Early Stopping
- Evaluación integral con métricas específicas para riesgo financiero

TÉCNICAS DE DEEP LEARNING IMPLEMENTADAS:
-------------------------------------------
- Embedding Layers: Representación densa de variables categóricas
- Mixed Precision Training: Eficiencia de memoria y velocidad
- OneCycleLR Scheduler: Convergencia superior y mejor generalización
- Regularización: BatchNorm + Dropout + Weight Decay
- F2-Score Optimization: Prioriza recall para minimizar falsos negativos

MÉTRICAS DE EVALUACIÓN FINANCIERA:
------------------------------------
- ROC-AUC: Capacidad discriminativa general del modelo
- PR-AUC: Rendimiento en clases desbalanceadas (crítico en riesgo)
- Brier Score: Calibración de probabilidades para toma de decisiones
- F2-Score: Métrica optimizada que prioriza recall sobre precisión

OPTIMIZACIONES DE RENDIMIENTO:
---------------------------------
- Monitorización continua de VRAM GPU durante entrenamiento
- DataLoader optimizado para Windows con multiprocessing
- Gradient Scaling automático para estabilidad numérica
- Early stopping inteligente basado en F2-score

ARTEFACTOS GENERADOS:
-----------------------
- best_model.pth: Pesos del modelo con mejor F2-score en validación
- history.json: Historial completo de métricas por época
- optuna_study.pkl: Estudio completo de optimización de hiperparámetros
- metrics.json: Métricas finales evaluadas en conjunto de prueba

Autor: Sistema de ML para Riesgo Crediticio
Fecha: Julio 2025
Versión: 2.0 - Optimizada para producción
"""

import warnings
# [CONFIGURACIÓN]: Supresión de warnings específicos de Optuna
# Optuna genera warnings sobre distribuciones que no afectan el entrenamiento
# pero pueden saturar los logs durante la optimización de hiperparámetros
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="optuna.distributions"
)

# 
# IMPORTS Y DEPENDENCIAS PRINCIPALES
# 

# [SERIALIZACIÓN]: Manejo de datos y persistencia de modelos
import json            # Serialización de métricas e histórico de entrenamiento
import joblib          # Persistencia del estudio de Optuna para análisis posterior
from pathlib import Path  # Manejo moderno de rutas de archivos

# [DEEP LEARNING]: Framework principal y utilidades
import torch           # Framework de Deep Learning con soporte GPU
import torch.multiprocessing as mp  # Multiprocesamiento optimizado para DataLoader
import numpy as np     # Operaciones numéricas vectorizadas y álgebra lineal
import pandas as pd    # Manipulación y análisis de datos tabulares

# [PYTORCH CORE]: Módulos esenciales para redes neuronales
from torch import nn                           # Capas y funciones de activación
from torch.utils.data import Dataset, DataLoader  # Carga eficiente de datos
from torch.cuda.amp import autocast, GradScaler   # Mixed precision para optimización

# [OPTIMIZACIÓN HPO]: Búsqueda automática de hiperparámetros
import optuna          # Framework para optimización bayesiana de hiperparámetros

# [MACHINE LEARNING]: Métricas y validación de modelos
from sklearn.model_selection import train_test_split  # División estratificada de datos
from sklearn.metrics import (                         # Suite completa de métricas
    roc_auc_score,           # Área bajo curva ROC (discriminación)
    average_precision_score, # Área bajo curva PR (clases desbalanceadas)
    brier_score_loss,        # Calibración de probabilidades
    accuracy_score,          # Precisión general del modelo
    precision_score,         # Precisión por clase (falsos positivos)
    recall_score,            # Sensibilidad (falsos negativos)
    f1_score,               # Media armónica de precisión y recall
    fbeta_score             # F-score parametrizable (priorizando recall)
)

# 
# CONFIGURACIÓN GLOBAL DEL SISTEMA
# 

# [REPRODUCIBILIDAD]: Semilla global para experimentos determinísticos
SEED         = 42    # Valor fijo para numpy, torch y división de datos

# [ENTRENAMIENTO]: Parámetros principales del pipeline
BATCH        = 2048  # Tamaño de lote: balance entre memoria GPU y convergencia
EPOCHS_FINAL = 15    # Número máximo de épocas en entrenamiento final
PATIENCE     = 3     # Épocas sin mejora antes de early stopping
BETA_F       = 2.0   # Factor beta para F2-score (prioriza recall sobre precisión)

# [HARDWARE]: Configuración automática de dispositivo de cómputo
CUDA   = torch.cuda.is_available()               # Detectar disponibilidad de GPU
DEVICE = torch.device("cuda" if CUDA else "cpu") # Seleccionar dispositivo óptimo

# [PERSISTENCIA]: Directorio para artefactos del modelo
REPORT_DIR = Path("reports")      # Carpeta de salida para todos los artefactos
REPORT_DIR.mkdir(exist_ok=True)   # Crear directorio si no existe

# [DETERMINISMO]: Configuración de semillas para reproducibilidad total
torch.manual_seed(SEED)    # Fijar semilla para operaciones de PyTorch
np.random.seed(SEED)       # Fijar semilla para operaciones de NumPy

# [MONITORIZACIÓN]: Si tienes GPU con múltiples procesos, también considera:
# torch.cuda.manual_seed(SEED) y torch.cuda.manual_seed_all(SEED)


def show_gpu(stage=""):
    """
    Monitor de uso de memoria GPU en tiempo real.
    
    Esta función proporciona información crítica sobre el consumo de VRAM
    durante las diferentes fases del entrenamiento, permitiendo optimizar
    el uso de memoria y detectar posibles memory leaks.
    
    Args:
        stage (str): Identificador de la fase actual del proceso
        
    Utilidad:
        Monitoreo continuo: Detecta picos de memoria durante el proceso
        Debug de memory leaks: Identifica incrementos anómalos de VRAM
        Optimización: Ayuda a ajustar BATCH_SIZE según hardware disponible
        
    Ejemplo de salida:
        [start] VRAM used: 0/11019 MiB
        [after model creation] VRAM used: 1250/11019 MiB
        [after training] VRAM used: 8950/11019 MiB
    """
    if CUDA:
        # Obtener memoria GPU utilizada en MiB (Mebibytes)
        used  = torch.cuda.memory_allocated() / 1024**2
        # Obtener memoria total disponible en la GPU
        total = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"[{stage}] VRAM used: {used:.0f}/{total:.0f} MiB")
    else:
        print(f"[{stage}] GPU not available - usando CPU")


# 
# DATASET PERSONALIZADO PARA DATOS DE RIESGO CREDITICIO
# 

class CreditDataset(Dataset):
    """
    Dataset optimizado para datos financieros con variables mixtas.
    
    Esta clase implementa un pipeline completo de preprocesamiento que combina
    técnicas específicas para variables numéricas y categóricas, optimizado
    para el dominio de riesgo crediticio.
    
    PIPELINE DE TRANSFORMACIÓN:
    ------------------------------
    1. Variables Numéricas → Normalización Z-score (μ=0, σ=1)
    2. Variables Categóricas → Mapping a índices enteros para embeddings
    3. Estructuras de datos → Tensores optimizados para entrenamiento GPU
    
    ESTRATEGIA DE EMBEDDINGS:
    ----------------------------
    Las variables categóricas se mapean a índices enteros que luego son 
    procesados por capas de embedding, permitiendo que el modelo aprenda
    representaciones densas y significativas de cada categoría.
    
    BENEFICIOS DEL ENFOQUE:
    --------------------------
    - Manejo eficiente de alta cardinalidad categórica
    - Preservación de relaciones semánticas entre categorías
    - Reducción de dimensionalidad vs one-hot encoding
    - Mejor generalización en datos no vistos
    
    Args:
        df (pd.DataFrame): DataFrame con características y variable objetivo
        num_cols (list): Lista de nombres de columnas numéricas
        cat_cols (list): Lista de nombres de columnas categóricas
        num_stats (tuple, optional): (means, stds) pre-calculados del conjunto de entrenamiento
        cat_maps (dict, optional): Mappings categoría→índice del conjunto de entrenamiento
        
    Returns:
        tuple: ((X_num, X_cat), y) donde:
            - X_num: tensor float32 con variables numéricas normalizadas
            - X_cat: tensor long con índices de variables categóricas
            - y: tensor float32 con etiquetas binarias (0=pagó, 1=impago)
            
    Ejemplo de uso:
        ```python
        # Crear dataset de entrenamiento (calcula estadísticos)
        train_ds = CreditDataset(train_df, num_cols, cat_cols)
        
        # Crear dataset de validación (reutiliza estadísticos)
        valid_ds = CreditDataset(
            valid_df, num_cols, cat_cols,
            num_stats=(train_ds.means, train_ds.stds),
            cat_maps=train_ds.cat_maps
        )
        ```
    """
    
    def __init__(self, df, num_cols, cat_cols, num_stats=None, cat_maps=None):
        
        # [ETIQUETAS]: Conversión a tensor float32 para función de pérdida
        # En riesgo crediticio: 0 = cliente pagó, 1 = cliente no pagó (default)
        self.y = torch.tensor(df["target"].values, dtype=torch.float32)
        
        # 
        # PROCESAMIENTO DE VARIABLES NUMÉRICAS
        # 
        
        if num_stats is None:
            # [ENTRENAMIENTO]: Calcular estadísticos sobre conjunto de entrenamiento
            # Media y desviación estándar para normalización Z-score
            self.means = df[num_cols].mean()  # Media por columna
            self.stds  = df[num_cols].std().replace(0, 1)  # Desvío (evitando división por 0)
        else:
            # [VALIDACIÓN/TEST]: Reutilizar estadísticos del entrenamiento
            # CRÍTICO: Usar las mismas estadísticas para evitar data leakage
            self.means, self.stds = num_stats
            
        # [NORMALIZACIÓN]: Aplicar transformación Z-score
        # Fórmula: (x - μ) / σ → distribución con media 0 y desviación 1
        # Beneficios: Convergencia más rápida, estabilidad numérica, gradientes balanceados
        normalized_data = (df[num_cols] - self.means) / self.stds
        self.X_num = torch.tensor(normalized_data.values, dtype=torch.float32)
        
        # 
        # PROCESAMIENTO DE VARIABLES CATEGÓRICAS
        # 
        
        if cat_maps is None:
            # [ENTRENAMIENTO]: Crear mapeos categoría → índice entero
            # Cada columna categórica obtiene su propio vocabulario de índices
            self.cat_maps = {
                col: {
                    category: idx 
                    for idx, category in enumerate(
                        df[col].astype('category').cat.categories
                    )
                }
                for col in cat_cols
            }
        else:
            # [VALIDACIÓN/TEST]: Reutilizar mapeos del entrenamiento
            # Esto garantiza consistencia en la codificación entre conjuntos
            self.cat_maps = cat_maps
            
        # [CODIFICACIÓN]: Convertir categorías a índices enteros
        # Proceso: string → índice → tensor long (requerido por nn.Embedding)
        categorical_data = []
        for col in cat_cols:
            # Mapear cada categoría a su índice correspondiente
            mapped_col = df[col].map(self.cat_maps[col])
            # Manejar categorías no vistas: asignar índice -1 (se puede manejar con embedding padding)
            mapped_col = mapped_col.fillna(-1).astype(int).values
            categorical_data.append(mapped_col)
            
        # [TENSOR]: Apilar columnas categóricas en tensor 2D
        # Shape: (n_samples, n_categorical_features)
        self.X_cat = torch.tensor(
            np.stack(categorical_data, axis=1), 
            dtype=torch.long  # Requerido por nn.Embedding
        )
    
    def __len__(self):
        """Retorna el número total de muestras en el dataset."""
        return len(self.y)
    
    def __getitem__(self, idx):
        """
        Retorna una muestra específica del dataset.
        
        Args:
            idx (int): Índice de la muestra a recuperar
            
        Returns:
            tuple: ((X_num, X_cat), y) estructurado para el modelo
                - X_num: Variables numéricas normalizadas
                - X_cat: Variables categóricas como índices
                - y: Etiqueta objetivo (0/1)
        """
        return (self.X_num[idx], self.X_cat[idx]), self.y[idx]


# 
# ARQUITECTURA DE RED NEURONAL HÍBRIDA PARA RIESGO CREDITICIO
# 

class RiskNN(nn.Module):
    """
    Red neuronal híbrida optimizada para datos financieros mixtos.
    
    Esta arquitectura combina embeddings densos para variables categóricas
    con procesamiento directo de variables numéricas, siguiendo las mejores
    prácticas de deep learning para datos tabulares.
    
    COMPONENTES DE LA ARQUITECTURA:
    ----------------------------------
    
    1. EMBEDDING LAYERS:
    • Transforman variables categóricas en representaciones densas
    • Dimensión automática: min(50, n_categories//2 + 1)
    • Permite capturar relaciones semánticas entre categorías
    
    2. BACKBONE DENSO:
    • Capas lineales con BatchNorm para estabilización
    • Activación GELU para mejor gradiente flow
    • Dropout progresivo para regularización
    
    3. CABEZA DE CLASIFICACIÓN:
    • Capa final a 1 neurona (logit crudo)
    • Sin sigmoid (se aplica en BCEWithLogitsLoss)
    
    DECISIONES DE DISEÑO JUSTIFICADAS:
    ------------------------------------
    
    - GELU vs ReLU: Mejor flujo de gradientes, menos "dead neurons"
    - BatchNorm: Estabiliza entrenamiento y acelera convergencia  
    - Dropout: Previene overfitting en datos financieros (alta varianza)
    - Embeddings: Más eficiente que one-hot para alta cardinalidad
    - Dimensionamiento automático: Balancea capacidad vs overfitting
    
    Args:
        num_features (int): Número de variables numéricas de entrada
        cat_dims (list[int]): Lista con número de categorías por variable categórica
        emb_dims (list[int]): Lista con dimensión de embedding por variable categórica
        hidden (tuple): Tuple con número de neuronas por capa oculta
        dropout (float): Probabilidad de dropout (0.1-0.5 recomendado)
        
    Arquitectura resultante:
        Input: (batch_size, num_features) + (batch_size, n_categoricals)
           ↓
        Embeddings: Cada categórica → embedding dense
           ↓  
        Concat: [embeddings] + [numéricas] → vector unificado
           ↓
        Dense Layers: Linear → BatchNorm → GELU → Dropout
           ↓
        Output: (batch_size, 1) logit crudo
        
    Ejemplo de uso:
        ```python
        model = RiskNN(
            num_features=15,           # 15 variables numéricas
            cat_dims=[10, 5, 3],      # 3 categóricas con 10, 5, 3 categorías
            emb_dims=[6, 3, 2],       # Embeddings de dim 6, 3, 2 respectivamente
            hidden=(256, 128),        # 2 capas ocultas de 256 y 128 neuronas
            dropout=0.3               # 30% dropout para regularización
        )
        ```
    """
    
    def __init__(self, num_features, cat_dims, emb_dims, hidden=(256,128), dropout=0.3):
        super().__init__()
        
        # 
        # CAPAS DE EMBEDDING PARA VARIABLES CATEGÓRICAS
        # 
        
        # [EMBEDDINGS]: Una capa de embedding por cada variable categórica
        # nn.Embedding(vocab_size, embedding_dim) crea una lookup table entrenable
        # que mapea índices enteros a vectores densos de dimensión fija
        self.emb = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim)
            for vocab_size, embed_dim in zip(cat_dims, emb_dims)
        ])
        
        # [INICIALIZACIÓN]: PyTorch inicializa embeddings con distribución normal
        # Para mejores resultados, se podría usar Xavier/He initialization
        
        # 
        # CONSTRUCCIÓN DEL BACKBONE DENSO
        # 
        
        # [DIMENSIONAMIENTO]: Calcular dimensión de entrada total
        # Suma de todas las dimensiones de embeddings + variables numéricas
        input_dim = num_features + sum(emb_dims)
        
        # [ARQUITECTURA]: Construcción dinámica de capas ocultas
        layers = []
        current_dim = input_dim
        
        for hidden_size in hidden:
            # [BLOQUE DENSO]: Patrón estándar para capas densas regularizadas
            layers.extend([
                # 1. Transformación lineal: y = xW^T + b
                nn.Linear(current_dim, hidden_size),
                
                # 2. Normalización de lote: estabiliza distribución de activaciones
                # Beneficios: convergencia más rápida, menos sensible a LR, regularización implícita
                nn.BatchNorm1d(hidden_size),
                
                # 3. Función de activación GELU: aproximación suave de ReLU
                # GELU(x) = x * Φ(x) donde Φ es CDF de distribución normal estándar
                # Ventajas vs ReLU: gradientes más suaves, menos "dead neurons"
                nn.GELU(),
                
                # 4. Dropout: regularización estocástica
                # Durante entrenamiento: establece aleatoriamente p% de neuronas a 0
                # Durante inferencia: escala activaciones por (1-p) para compensar
                nn.Dropout(dropout)
            ])
            current_dim = hidden_size  # Actualizar dimensión para siguiente capa
        
        # [CABEZA DE CLASIFICACIÓN]: Capa final sin activación
        # Salida: logit crudo (real number) que será procesado por BCEWithLogitsLoss
        # No se aplica sigmoid aquí por estabilidad numérica de la función de pérdida
        layers.append(nn.Linear(current_dim, 1))
        
        # [SECUENCIAL]: Empaquetar todas las capas en un módulo secuencial
        self.net = nn.Sequential(*layers)
    
    def forward(self, x_num, x_cat):
        """
        Forward pass de la red neuronal híbrida.
        
        Este método implementa el flujo de datos desde las entradas brutas
        hasta el logit de salida, combinando embeddings categóricos con
        características numéricas.
        
        Args:
            x_num (torch.Tensor): Variables numéricas normalizadas
                Shape: (batch_size, num_features)
            x_cat (torch.Tensor): Índices de variables categóricas  
                Shape: (batch_size, n_categorical_features)
                
        Returns:
            torch.Tensor: Logits de clasificación binaria
                Shape: (batch_size,) - vector 1D de logits
                
        Flujo de procesamiento:
            Input → Embeddings → Concatenación → Backbone → Output
        """
        
        # 
        # 1. GENERACIÓN DE EMBEDDINGS CATEGÓRICOS
        # 
        
        # [EMBEDDINGS]: Convertir cada índice categórico a su embedding correspondiente
        # x_cat[:, i] extrae la i-ésima columna categórica (todos los samples)
        # self.emb[i] es la i-ésima capa de embedding
        # Resultado: lista de tensores, cada uno con shape (batch_size, embed_dim_i)
        embeddings = [
            embedding_layer(x_cat[:, col_idx]) 
            for col_idx, embedding_layer in enumerate(self.emb)
        ]
        
        # [DEBUG INFO]: En desarrollo, puedes verificar shapes:
        # print(f"Embeddings shapes: {[emb.shape for emb in embeddings]}")
        # print(f"Numerical features shape: {x_num.shape}")
        
        # 
        # 2. CONCATENACIÓN DE CARACTERÍSTICAS
        # 
        
        # [FUSIÓN]: Combinar embeddings categóricos + características numéricas
        # torch.cat concatena tensores a lo largo de la dimensión especificada
        # dim=1 → concatenar a lo largo del eje de características (columnas)
        # Resultado: tensor unificado con todas las características
        unified_features = torch.cat(embeddings + [x_num], dim=1)
        
        # [SHAPE FINAL]: (batch_size, total_features)
        # donde total_features = sum(emb_dims) + num_features
        
        # 
        # 3. PROCESAMIENTO A TRAVÉS DEL BACKBONE
        # 
        
        # [PREDICCIÓN]: Pasar características unificadas por la red densa
        # self.net aplica secuencialmente: Linear → BatchNorm → GELU → Dropout → ... → Linear
        raw_output = self.net(unified_features)
        
        # [RESHAPE]: Aplanar de (batch_size, 1) a (batch_size,)
        # squeeze(1) elimina la dimensión 1 (segunda dimensión)
        # Esto es requerido por BCEWithLogitsLoss que espera vector 1D
        logits = raw_output.squeeze(1)
        
        return logits
        
        # [NOTA]: El sigmoid se aplica internamente en BCEWithLogitsLoss
        # Para obtener probabilidades en inferencia: torch.sigmoid(logits)


# 
# FUNCIONES DE ENTRENAMIENTO Y EVALUACIÓN AVANZADAS
# 

def train_one_epoch(model, loader, criterion, optimizer, scaler, scheduler=None):
    """
    Entrena el modelo durante una época completa con técnicas SOTA.
    
    Esta función implementa un loop de entrenamiento optimizado que combina
    multiple técnicas avanzadas para maximizar eficiencia y estabilidad:
    
    TÉCNICAS IMPLEMENTADAS:
    -------------------------
    • Mixed Precision Training: Reduce uso de VRAM ~50% y acelera entrenamiento
    • Gradient Scaling: Mantiene precisión numérica en operaciones float16
    • OneCycleLR Integration: Scheduler que mejora convergencia y generalización
    • Memory-Efficient Batching: Optimización de transferencias GPU
    
    BENEFICIOS DE RENDIMIENTO:
    ---------------------------
    • VRAM: Reducción significativa de memoria GPU vs float32 puro
    • Velocidad: 1.5-2x más rápido en GPUs modernas (V100, RTX, A100)
    • Convergencia: Mejor estabilidad numérica con gradient scaling
    • Generalización: OneCycleLR mejora capacidad de generalización
    
    Args:
        model (nn.Module): Red neuronal a entrenar
        loader (DataLoader): DataLoader con datos de entrenamiento
        criterion (nn.Module): Función de pérdida (recomendado: BCEWithLogitsLoss)
        optimizer (torch.optim): Optimizador (recomendado: AdamW)
        scaler (GradScaler): Escalador de gradientes para mixed precision
        scheduler (optional): Scheduler de learning rate (recomendado: OneCycleLR)
        
    Returns:
        float: Pérdida promedio de la época, normalizada por número total de samples
        
    Notas técnicas:
        Mixed precision usa float16 en forward pass, float32 en backward pass
        Gradient scaling multiplica pérdida por ~65536 para evitar underflow
        OneCycleLR requiere step() en cada batch, no al final de época
    """
    
    # [MODO ENTRENAMIENTO]: Habilitar dropout, batch norm en modo training
    model.train()
    total_loss = 0.0
    
    # [LOOP PRINCIPAL]: Iterar sobre batches del conjunto de entrenamiento
    for (x_num, x_cat), y in loader:
        
        # 
        # 1. TRANSFERENCIA DE DATOS A GPU
        # 
        
        # [OPTIMIZACIÓN]: Mover tensores a GPU antes del forward pass
        # Esto minimiza las transferencias CPU→GPU durante el entrenamiento
        # y mantiene todos los datos en la misma ubicación de memoria
        x_num = x_num.to(DEVICE, non_blocking=True)  # Variables numéricas
        x_cat = x_cat.to(DEVICE, non_blocking=True)  # Variables categóricas  
        y = y.to(DEVICE, non_blocking=True)          # Etiquetas objetivo
        
        # 
        # 2. LIMPIEZA DE GRADIENTES ACUMULADOS
        # 
        
        # [CRÍTICO]: Limpiar gradientes del batch anterior
        # PyTorch acumula gradientes por defecto, por lo que debemos
        # limpiarlos explícitamente antes de cada backward pass
        optimizer.zero_grad()
        
        # 
        # 3. FORWARD PASS CON MIXED PRECISION
        # 
        
        # [MIXED PRECISION]: Forward pass en precisión mixta para eficiencia
        # autocast() automáticamente selecciona float16/float32 por operación:
        # - Operaciones "seguras" (matmul, conv) → float16 (más rápido)
        # - Operaciones "peligrosas" (softmax, loss) → float32 (más estable)
        with autocast():
            # [PREDICCIÓN]: Obtener logits crudos del modelo
            logits = model(x_num, x_cat)
            
            # [PÉRDIDA]: Calcular BCE loss con balanceo de clases
            # BCEWithLogitsLoss = sigmoid + BCE en una sola operación
            # Más estable numéricamente que aplicar sigmoid por separado
            loss = criterion(logits, y)
        
        # 
        # 4. BACKWARD PASS CON GRADIENT SCALING
        # 
        
        # [GRADIENT SCALING]: Proceso de 3 pasos para mantener precisión
        # 1. Escalar pérdida por factor grande (~65536) antes de backward
        scaler.scale(loss).backward()
        
        # 2. Desescalar gradientes y actualizar parámetros
        scaler.step(optimizer)
        
        # 3. Actualizar el factor de escala para próxima iteración
        scaler.update()
        
        # [EXPLICACIÓN]: El scaling previene que gradientes pequeños
        # se conviertan en 0 en float16, manteniendo precisión del entrenamiento
        
        # 
        # 5. ACTUALIZACIÓN DE LEARNING RATE
        # 
        
        # [SCHEDULER]: Actualizar learning rate si se proporciona scheduler
        # OneCycleLR requiere step() en cada batch (no al final de época)
        # Esto implementa el ciclo: warm-up → peak LR → cool-down
        if scheduler is not None:
            scheduler.step()
        
        # 
        # 6. ACUMULACIÓN DE MÉTRICAS
        # 
        
        # [MÉTRICA]: Acumular pérdida ponderada por tamaño de batch
        # .item() extrae valor escalar del tensor (evita acumulación de gradientes)
        # Multiplicamos por batch_size para ponderar correctamente al promediar
        batch_loss = loss.item() * y.size(0)
        total_loss += batch_loss
    
    # [NORMALIZACIÓN]: Calcular pérdida promedio sobre toda la época
    # Dividimos por número total de samples para obtener pérdida promedio
    average_loss = total_loss / len(loader.dataset)
    
    return average_loss


@torch.no_grad()
def evaluate_probs(model, loader):
    """
    Evalúa modelo y retorna probabilidades para análisis de métricas.
    
    Esta función realiza inferencia eficiente sobre un conjunto de datos
    y retorna tanto las etiquetas verdaderas como las probabilidades predichas,
    optimizada para evaluación posterior de múltiples métricas.
    
    OPTIMIZACIONES IMPLEMENTADAS:
    -------------------------------
    • @torch.no_grad(): Desactiva cálculo de gradientes (menos memoria, más rápido)
    • model.eval(): Desactiva dropout y usa estadísticas fijas de batch norm
    • Batch processing: Procesa datos en lotes para eficiencia de GPU
    • CPU transfer: Mueve resultados a CPU para liberación de VRAM
    
    Args:
        model (nn.Module): Modelo entrenado para evaluación
        loader (DataLoader): DataLoader con datos de validación/test
        
    Returns:
        tuple[np.ndarray, np.ndarray]: (etiquetas_verdaderas, probabilidades_predichas)
            - etiquetas_verdaderas: Array 1D con valores binarios {0, 1}
            - probabilidades_predichas: Array 1D con probabilidades [0, 1]
            
    Uso típico:
        ```python
        y_true, y_prob = evaluate_probs(model, test_loader)
        metrics = evaluate_metrics(y_true, y_prob)
        ```
    """
    
    # [MODO EVALUACIÓN]: Configurar modelo para inferencia
    # - Dropout → desactivado (comportamiento determinístico)
    # - BatchNorm → usa estadísticas acumuladas (no del batch actual)
    model.eval()
    
    # [ACUMULADORES]: Listas para recopilar resultados por batch
    true_labels = []      # Etiquetas verdaderas de cada batch
    probabilities = []    # Probabilidades predichas de cada batch
    
    # [LOOP DE INFERENCIA]: Procesar cada batch sin calcular gradientes
    for (x_num, x_cat), y in loader:
        
        # [GPU TRANSFER]: Mover características a dispositivo de cómputo
        x_num = x_num.to(DEVICE, non_blocking=True)
        x_cat = x_cat.to(DEVICE, non_blocking=True)
        
        # [PREDICCIÓN]: Forward pass para obtener logits
        logits = model(x_num, x_cat)
        
        # [PROBABILIDADES]: Convertir logits a probabilidades con sigmoid
        # sigmoid(x) = 1 / (1 + exp(-x)) → mapea (-∞, +∞) a [0, 1]
        batch_probabilities = torch.sigmoid(logits)
        
        # [ACUMULACIÓN]: Guardar resultados del batch
        # .cpu() mueve tensores de GPU a CPU para liberar VRAM
        # .numpy() convierte tensores a arrays de NumPy para análisis posterior
        true_labels.append(y.cpu().numpy())
        probabilities.append(batch_probabilities.cpu().numpy())
    
    # [CONCATENACIÓN]: Unir todos los batches en arrays únicos
    # np.concatenate une arrays a lo largo del primer eje (samples)
    all_true_labels = np.concatenate(true_labels)
    all_probabilities = np.concatenate(probabilities)
    
    return all_true_labels, all_probabilities


def evaluate_metrics(y_true, y_prob):
    """
    Calcula suite completa de métricas para evaluación de riesgo crediticio.
    
    Esta función computa un conjunto integral de métricas específicamente
    seleccionadas para evaluar modelos de riesgo crediticio, cada una
    proporcionando insights únicos sobre el rendimiento del modelo.
    
    MÉTRICAS IMPLEMENTADAS Y SU RELEVANCIA:
    -----------------------------------------
    
    ROC-AUC (Receiver Operating Characteristic):
    • Mide capacidad discriminativa general del modelo
    • Independiente del threshold de clasificación
    • Valor: 0.5 = aleatorio, 1.0 = perfecto
    • Relevancia: Capacidad general de distinguir pagadores vs morosos
    
    PR-AUC (Precision-Recall Area Under Curve):
    • Crítico para datasets desbalanceados (pocos morosos vs muchos pagadores)
    • Más sensible a la clase minoritaria que ROC-AUC
    • Valor: baseline = prevalencia de clase positiva
    • Relevancia: Rendimiento específico en detección de morosos
    
    Brier Score (Calibración de probabilidades):
    • Mide qué tan bien calibradas están las probabilidades predichas
    • Formula: mean((probabilidad - realidad)²)
    • Valor: 0 = calibración perfecta, 0.25 = sin habilidad predictiva
    • Relevancia: Confiabilidad para decisiones basadas en probabilidad
    
    Métricas de Clasificación (threshold = 0.5):
    • Accuracy: Proporción total de predicciones correctas
    • Precision: De los predichos como morosos, % que realmente lo son
    • Recall: De los morosos reales, % que el modelo detecta
    • F1: Media armónica de precision y recall
    • F2: F-beta con β=2, prioriza recall sobre precision
    
    ¿Por qué F2-Score para Early Stopping?
    En riesgo crediticio, es MÁS COSTOSO no detectar un moroso (falso negativo)
    que rechazar incorrectamente un buen cliente (falso positivo).
    F2 penaliza más los falsos negativos, alineándose con objetivos de negocio.
    
    Args:
        y_true (np.ndarray): Etiquetas verdaderas binarias {0, 1}
        y_prob (np.ndarray): Probabilidades predichas [0, 1]
        
    Returns:
        dict: Diccionario con todas las métricas calculadas
        
    Ejemplo de interpretación:
        metrics = {
            "roc": 0.85,      # Buena capacidad discriminativa
            "pr": 0.45,       # Razonable para clase desbalanceada  
            "brier": 0.15,    # Probabilidades bien calibradas
            "accuracy": 0.82, # 82% de predicciones correctas
            "precision": 0.35,# 35% de "morosos predichos" son realmente morosos
            "recall": 0.78,   # 78% de morosos reales son detectados
            "f1": 0.48,       # Balance entre precision y recall
            "f2": 0.65        # Priorizando detección de morosos
        }
    """
    
    # 
    # MÉTRICAS INDEPENDIENTES DE THRESHOLD
    # 
    
    # [ROC-AUC]: Área bajo curva ROC (sensibilidad vs 1-especificidad)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    # [PR-AUC]: Área bajo curva Precision-Recall
    # Especialmente importante cuando tenemos clases desbalanceadas
    pr_auc = average_precision_score(y_true, y_prob)
    
    # [BRIER SCORE]: Medida de calibración de probabilidades
    # Brier = mean((prob - real)²) → menor es mejor
    brier = brier_score_loss(y_true, y_prob)
    
    # 
    # MÉTRICAS BASADAS EN THRESHOLD FIJO (0.5)
    # 
    
    # [BINARIZACIÓN]: Convertir probabilidades a predicciones binarias
    # Threshold = 0.5: probabilidad ≥ 0.5 → clase positiva (moroso)
    y_pred = (y_prob >= 0.5).astype(int)
    
    # [MÉTRICAS DE CLASIFICACIÓN]: Calcular métricas estándar
    accuracy = accuracy_score(y_true, y_pred)
    
    # zero_division=0: retorna 0 si denominador es 0 (evita errores)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # [F2-SCORE]: F-beta con β=2 para priorizar recall
    # F_β = (1 + β²) * (precision * recall) / (β² * precision + recall)
    # β=2 significa que recall es 2x más importante que precision
    f2 = fbeta_score(y_true, y_pred, beta=BETA_F, zero_division=0)
    
    # [RESULTADO]: Diccionario con todas las métricas organizadas
    return {
        # Métricas independientes de threshold
        "roc": roc_auc,
        "pr": pr_auc, 
        "brier": brier,
        
        # Métricas de clasificación binaria
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2  # Métrica principal para early stopping
    }


# 
# PIPELINE PRINCIPAL: ENTRENAMIENTO COMPLETO DE RIESGO CREDITICIO
# 

def main():
    """
    Pipeline completo de Machine Learning para riesgo crediticio.
    
    Esta función orquesta todo el proceso de entrenamiento, desde la carga
    de datos hasta la evaluación final, implementando las mejores prácticas
    de MLOps y optimización de hiperparámetros.
    
    FASES DEL PIPELINE:
    ---------------------
    1. Configuración del entorno y carga de datos
    2. Preprocesamiento y división estratificada
    3. Creación de datasets y loaders optimizados
    4. Optimización automática de hiperparámetros (Optuna)
    5. Entrenamiento final con mejores hiperparámetros
    6. Evaluación en conjunto de prueba y persistencia
    
    ARTEFACTOS GENERADOS:
    -----------------------
    • best_model.pth: Pesos del mejor modelo (F2-score)
    • history.json: Historial completo de entrenamiento
    • optuna_study.pkl: Estudio de optimización para análisis
    • metrics.json: Métricas finales en conjunto de prueba
    
    MÉTRICAS DE ÉXITO:
    --------------------
    • ROC-AUC > 0.75: Buena capacidad discriminativa
    • F2-Score > 0.60: Balance adecuado priorizando recall
    • Brier Score < 0.20: Probabilidades bien calibradas
    """
    
    # 
    # CONFIGURACIÓN INICIAL DEL ENTORNO
    # 
    
    # [WINDOWS FIX]: Configurar multiprocessing para DataLoader
    # En Windows, el método por defecto 'fork' no está disponible
    # 'spawn' inicia procesos completamente nuevos, evitando errores
    mp.set_start_method('spawn', force=True)
    
    # [MONITOREO]: Baseline de uso de memoria GPU antes de cargar datos
    show_gpu("Inicio del pipeline")
    
    # 
    # CARGA Y LIMPIEZA DE DATOS
    # 
    
    # [CARGA]: Leer dataset procesado desde CSV
    # Este archivo debe contener datos ya limpios y con feature engineering
    df = pd.read_csv("data/processed/data_loan_complete.csv")
    
    # [LIMPIEZA]: Filtrar registros con etiquetas faltantes
    # En problemas de clasificación, necesitamos etiquetas válidas para supervisión
    initial_size = len(df)
    df = df[df["loan_status_bin"].notna()].copy()
    final_size = len(df)
    print(f"Datos cargados: {initial_size:,} → {final_size:,} registros válidos")
    
    # [ETIQUETAS]: Crear variable objetivo binaria
    # 0 = cliente pagó el préstamo (clase negativa)
    # 1 = cliente no pagó el préstamo - default (clase positiva)
    df["target"] = df["loan_status_bin"].astype(int)
    
    # [LIMPIEZA]: Eliminar columna original para evitar data leakage
    df.drop(columns=["loan_status_bin"], inplace=True)
    
    # [ANÁLISIS]: Mostrar distribución de clases
    class_counts = df["target"].value_counts()
    class_0_pct = (class_counts[0] / len(df)) * 100
    class_1_pct = (class_counts[1] / len(df)) * 100
    print(f"Distribución de clases:")
    print(f"   • Clase 0 (pagaron): {class_counts[0]:,} ({class_0_pct:.1f}%)")
    print(f"   • Clase 1 (morosos): {class_counts[1]:,} ({class_1_pct:.1f}%)")
    
    # 
    # DIVISIÓN ESTRATIFICADA DE DATOS
    # 
    
    # [ESTRATEGIA]: División 70% train, 15% validation, 15% test
    # Estratificada: mantiene proporción de clases en cada conjunto
    
    # Primera división: separar conjunto de prueba (15%)
    train_df, test_df = train_test_split(
        df,
        test_size=0.15,           # 15% para test
        stratify=df["target"],    # Mantener proporción de clases
        random_state=SEED         # Reproducibilidad
    )
    
    # Segunda división: separar validación del entrenamiento
    # 0.1765 ≈ 15% del total original (15% / 85% = 0.1765)
    train_df, valid_df = train_test_split(
        train_df,
        test_size=0.1765,         # 15% del total original
        stratify=train_df["target"],  # Mantener proporción de clases
        random_state=SEED         # Reproducibilidad
    )
    
    # [VERIFICACIÓN]: Mostrar tamaños finales de conjuntos
    print(f"División de datos:")
    print(f"   • Entrenamiento: {len(train_df):,} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   • Validación: {len(valid_df):,} samples ({len(valid_df)/len(df)*100:.1f}%)")
    print(f"   • Prueba: {len(test_df):,} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    show_gpu("Después de división de datos")
    
    # 
    # IDENTIFICACIÓN DE TIPOS DE VARIABLES
    # 
    
    # [CATEGÓRICAS]: Identificar columnas con dtype 'object' (strings)
    categorical_cols = [col for col in df.columns if df[col].dtype == "object"]
    
    # [NUMÉRICAS]: Todo lo que no sea categórico ni la variable objetivo
    numerical_cols = [
        col for col in df.columns 
        if col not in categorical_cols + ["target"]
    ]
    
    print(f"Análisis de características:")
    print(f"   • Variables numéricas: {len(numerical_cols)}")
    print(f"   • Variables categóricas: {len(categorical_cols)}")
    
    # 
    # DIMENSIONAMIENTO AUTOMÁTICO DE EMBEDDINGS
    # 
    
    # [CARDINALIDAD]: Número de categorías únicas por variable categórica
    cat_dims = [train_df[col].nunique() for col in categorical_cols]
    
    # [EMBEDDINGS]: Dimensión de embedding por variable categórica
    # Fórmula heurística: min(50, n_categories//2 + 1)
    # - Evita embeddings demasiado grandes (máximo 50)
    # - Escala con cardinalidad pero no linealmente
    # - +1 asegura dimensión mínima de 1 para cardinalidad baja
    emb_dims = [min(50, dim//2 + 1) for dim in cat_dims]
    
    print(f"Configuración de embeddings:")
    for col, card, emb_dim in zip(categorical_cols, cat_dims, emb_dims):
        print(f"   • {col}: {card} categorías → embedding dim {emb_dim}")

    
    # 
    # CREACIÓN DE DATASETS Y DATALOADERS OPTIMIZADOS
    # 
    
    #  [DATASET ENTRENAMIENTO]: Calcula estadísticos de normalización y mapeos
    train_ds = CreditDataset(train_df, numerical_cols, categorical_cols)
    
    #  [DATASET VALIDACIÓN]: Reutiliza estadísticos del entrenamiento
    # CRÍTICO: Evita data leakage usando las mismas transformaciones
    valid_ds = CreditDataset(
        valid_df, numerical_cols, categorical_cols,
        num_stats=(train_ds.means, train_ds.stds),  # Reutilizar normalización
        cat_maps=train_ds.cat_maps                  # Reutilizar mapeos categóricos
    )
    
    #  [DATASET PRUEBA]: Mismo patrón para conjunto de prueba final
    test_ds = CreditDataset(
        test_df, numerical_cols, categorical_cols,
        num_stats=(train_ds.means, train_ds.stds),
        cat_maps=train_ds.cat_maps
    )
    
    #  [DATALOADERS]: Configuración optimizada para GPU + Windows
    # num_workers=0: Evita problemas de multiprocessing en Windows
    # pin_memory=False: Más estable en configuraciones mixtas CPU/GPU
    # shuffle=True: Solo en entrenamiento para variar orden de batches
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH,
        shuffle=True,        #  Aleatorizar orden en entrenamiento
        num_workers=0,       #  Evita problemas en Windows
        pin_memory=False     #  Más estable para desarrollo
    )
    
    valid_loader = DataLoader(
        valid_ds,
        batch_size=BATCH,
        shuffle=False,       #  Orden determinístico para evaluación
        num_workers=0,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH, 
        shuffle=False,       #  Orden determinístico para prueba final
        num_workers=0,
        pin_memory=False
    )
    
    print(f" DataLoaders creados:")
    print(f"   • Entrenamiento: {len(train_loader)} batches de {BATCH}")
    print(f"   • Validación: {len(valid_loader)} batches de {BATCH}")
    print(f"   • Prueba: {len(test_loader)} batches de {BATCH}")
    
    # 
    #  OPTIMIZACIÓN AUTOMÁTICA DE HIPERPARÁMETROS CON OPTUNA
    # 
    
    def objective(trial):
        """
         Función objetivo para optimización bayesiana de hiperparámetros.
        
        Esta función define el espacio de búsqueda de hiperparámetros y
        entrena un modelo por tiempo limitado para evaluar su rendimiento.
        Optuna usa los resultados para guiar la búsqueda hacia regiones
        prometedoras del espacio de hiperparámetros.
        
         ESPACIO DE BÚSQUEDA JUSTIFICADO:
        ----------------------------------
        • hidden: Arquitecturas probadas en literatura de tabular data
        • dropout: Rango que balancea regularización vs capacidad
        • lr: Rango logarítmico centrado en valores típicos para AdamW
        • wd: Weight decay en rango que previene overfitting sin dañar convergencia
        
        Args:
            trial (optuna.Trial): Objeto trial de Optuna para sugerir hiperparámetros
            
        Returns:
            float: F2-score máximo alcanzado durante el entrenamiento corto
        """
        
        # 
        #  DEFINICIÓN DEL ESPACIO DE BÚSQUEDA
        # 
        
        #  [ARQUITECTURA]: Configuraciones probadas en deep learning tabular
        hidden = trial.suggest_categorical(
            "hidden", 
            [
                (256, 128),      #  Configuración estándar: decrece gradualmente
                (128, 64),       #  Configuración ligera: más rápida, menos overfitting
                (256, 128, 64)   #  Configuración profunda: mayor capacidad
            ]
        )
        
        #  [REGULARIZACIÓN]: Dropout rate en rango efectivo
        # 0.1-0.5: rango que balancea regularización vs capacidad del modelo
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        
        #  [LEARNING RATE]: Búsqueda logarítmica centrada en valores óptimos
        # 1e-4 a 5e-3: rango típico para AdamW en problemas de clasificación
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        
        #  [WEIGHT DECAY]: Regularización L2 para prevenir overfitting
        # 1e-6 a 1e-3: rango que previene overfitting sin dañar convergencia
        wd = trial.suggest_float("wd", 1e-6, 1e-3, log=True)
        
        # 
        #  BALANCEO DE CLASES PARA BCE LOSS
        # 
        
        #  [CONTEO]: Contar muestras por clase en entrenamiento
        class_counts = np.bincount(train_df["target"].astype(int))
        c0, c1 = class_counts[0], class_counts[1]  # Pagadores, Morosos
        
        #  [PESO]: Calcular peso para balancear clases en BCE loss
        # pos_weight = n_negative / n_positive
        # Penaliza más los errores en la clase minoritaria (morosos)
        pos_weight = torch.tensor([c0/c1], device=DEVICE)
        
        print(f" Trial {trial.number}: Balance de clases → pos_weight = {c0/c1:.2f}")
        
        # 
        #  CONSTRUCCIÓN DEL MODELO Y OPTIMIZADORES
        # 
        
        #  [MODELO]: Instanciar red con hiperparámetros del trial
        model = RiskNN(
            num_features=len(numerical_cols),
            cat_dims=cat_dims,
            emb_dims=emb_dims,
            hidden=hidden,
            dropout=dropout
        ).to(DEVICE)
        
        #  [PÉRDIDA]: BCE con balanceo automático de clases
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        #  [OPTIMIZADOR]: AdamW con weight decay para regularización
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=wd
        )
        
        #  [MIXED PRECISION]: Scaler para entrenamiento en float16/float32
        scaler = GradScaler()
        
        #  [SCHEDULER]: OneCycleLR para 5 épocas de exploración rápida
        # total_steps = batches_por_época * número_de_épocas
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=len(train_loader) * 5  # 5 épocas de evaluación rápida
        )
        
        # 
        #  ENTRENAMIENTO RÁPIDO PARA EVALUACIÓN (5 ÉPOCAS)
        # 
        
        best_f2 = 0.0  # Mejor F2-score alcanzado en este trial
        
        #  [LOOP RÁPIDO]: 5 épocas para evaluar potencial del conjunto de HP
        for epoch in range(5):
            
            #  [ENTRENAMIENTO]: Una época completa
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, scheduler
            )
            
            #  [EVALUACIÓN]: Calcular métricas en validación
            y_true_val, y_prob_val = evaluate_probs(model, valid_loader)
            metrics = evaluate_metrics(y_true_val, y_prob_val)
            
            #  [LOGGING]: Mostrar progreso del trial
            print(
                f" Trial {trial.number:2d} Época {epoch+1}/5 → "
                f"Loss={train_loss:.4f} | ROC={metrics['roc']:.4f} | "
                f"PR={metrics['pr']:.4f} | Brier={metrics['brier']:.4f} | "
                f"Acc={metrics['accuracy']:.4f} | Prec={metrics['precision']:.4f} | "
                f"Rec={metrics['recall']:.4f} | F1={metrics['f1']:.4f} | "
                f"F2={metrics['f2']:.4f}"
            )
            
            #  [OPTUNA]: Reportar métrica intermedia para pruning
            trial.report(metrics['f2'], epoch)
            
            #  [PRUNING]: Terminar trial temprano si no es prometedor
            # Optuna compara con otros trials y detiene los que van mal
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            #  [TRACKING]: Actualizar mejor F2-score del trial
            best_f2 = max(best_f2, metrics['f2'])
        
        return best_f2
    
    # 
    #  EJECUCIÓN DE LA OPTIMIZACIÓN BAYESIANA
    # 
    
    print("\n Iniciando optimización de hiperparámetros con Optuna...")
    
    #  [ESTUDIO]: Configurar optimización bayesiana con Optuna
    study = optuna.create_study(
        direction="maximize",  #  Maximizar F2-score
        
        #  [SAMPLER]: TPE (Tree-structured Parzen Estimator)
        # Algoritmo bayesiano que modela P(x|y) para guiar búsqueda
        # Más eficiente que grid search o random search
        sampler=optuna.samplers.TPESampler(seed=SEED),
        
        #  [PRUNER]: MedianPruner para terminar trials poco prometedores
        # n_warmup_steps=1: permite al menos 1 época antes de pruning
        # Ahorra tiempo computacional eliminando configuraciones malas temprano
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1)
    )
    
    #  [OPTIMIZACIÓN]: Ejecutar 20 trials de búsqueda
    # Balance entre exploración completa y tiempo computacional
    study.optimize(objective, n_trials=20)
    
    #  [PERSISTENCIA]: Guardar estudio completo para análisis posterior
    joblib.dump(study, REPORT_DIR / "optuna_study.pkl")
    
    #  [MEJORES RESULTADOS]: Mostrar hiperparámetros óptimos encontrados
    best_params = study.best_trial.params
    best_f2_hpo = study.best_value
    
    print(f"\n Mejores hiperparámetros encontrados:")
    print(f"   • Arquitectura: {best_params['hidden']}")
    print(f"   • Dropout: {best_params['dropout']:.3f}")
    print(f"   • Learning Rate: {best_params['lr']:.6f}")
    print(f"   • Weight Decay: {best_params['wd']:.6f}")
    print(f"   • Mejor F2-Score: {best_f2_hpo:.4f}")
    
    show_gpu(" Después de optimización HPO")

    
    # 
    #  ENTRENAMIENTO FINAL CON HIPERPARÁMETROS ÓPTIMOS
    # 
    
    print(f"\n Iniciando entrenamiento final con mejores hiperparámetros...")
    
    #  [HIPERPARÁMETROS]: Usar configuración óptima encontrada por Optuna
    best_params = study.best_trial.params
    
    #  [BALANCEO]: Recalcular pos_weight para el modelo final
    class_counts = np.bincount(train_df["target"].astype(int))
    c0, c1 = class_counts[0], class_counts[1]
    pos_weight = torch.tensor([c0/c1], device=DEVICE)
    
    #  [MODELO FINAL]: Instanciar con hiperparámetros óptimos
    final_model = RiskNN(
        num_features=len(numerical_cols),
        cat_dims=cat_dims,
        emb_dims=emb_dims,
        hidden=best_params["hidden"],
        dropout=best_params["dropout"]
    ).to(DEVICE)
    
    #  [COMPONENTES]: Criterio, optimizador y scaler con configuración final
    final_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    final_optimizer = torch.optim.AdamW(
        final_model.parameters(),
        lr=best_params["lr"],
        weight_decay=best_params["wd"]
    )
    final_scaler = GradScaler()
    
    #  [SCHEDULER]: OneCycleLR para todo el entrenamiento final
    final_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        final_optimizer,
        max_lr=best_params["lr"],
        total_steps=len(train_loader) * EPOCHS_FINAL
    )
    
    #  [HISTÓRICO]: Diccionario para guardar métricas por época
    training_history = {
        "train_loss": [],
        "valid_loss": []
    }
    
    # Agregar placeholders para todas las métricas de evaluación
    metric_names = ["roc", "pr", "brier", "accuracy", "precision", "recall", "f1", "f2"]
    for metric_name in metric_names:
        training_history[metric_name] = []
    
    #  [EARLY STOPPING]: Variables para control de parada temprana
    best_f2_final = 0.0
    patience_counter = 0
    
    show_gpu(" Antes del entrenamiento final")
    
    # 
    #  LOOP PRINCIPAL DE ENTRENAMIENTO FINAL
    # 
    
    print(f"\n Entrenando modelo final por hasta {EPOCHS_FINAL} épocas...")
    print(f"⏰ Early stopping: paciencia de {PATIENCE} épocas en F2-score\n")
    
    for epoch in range(1, EPOCHS_FINAL + 1):
        
        # 
        #  FASE DE ENTRENAMIENTO
        # 
        
        #  [ENTRENAMIENTO]: Ejecutar una época completa
        train_loss = train_one_epoch(
            final_model, train_loader, final_criterion,
            final_optimizer, final_scaler, final_scheduler
        )
        
        # 
        #  FASE DE EVALUACIÓN EN VALIDACIÓN
        # 
        
        #  [EVALUACIÓN]: Obtener predicciones en conjunto de validación
        y_true_val, y_prob_val = evaluate_probs(final_model, valid_loader)
        validation_metrics = evaluate_metrics(y_true_val, y_prob_val)
        
        #  [VALIDATION LOSS]: Calcular pérdida en validación usando GPU
        # Esto es más eficiente que recalcular con evaluate_probs
        with torch.no_grad():
            val_logits_list = []
            val_targets_list = []
            
            for (x_num, x_cat), y in valid_loader:
                x_num = x_num.to(DEVICE, non_blocking=True)
                x_cat = x_cat.to(DEVICE, non_blocking=True)
                
                val_logits_list.append(final_model(x_num, x_cat))
                val_targets_list.append(y.to(DEVICE, non_blocking=True))
            
            # Concatenar todos los logits y targets
            all_val_logits = torch.cat(val_logits_list)
            all_val_targets = torch.cat(val_targets_list)
            
            # Calcular pérdida de validación
            validation_loss = final_criterion(all_val_logits, all_val_targets).item()
        
        # 
        #  GUARDADO DE HISTÓRICO Y MÉTRICAS
        # 
        
        #  [HISTÓRICO]: Guardar métricas de la época
        training_history["train_loss"].append(train_loss)
        training_history["valid_loss"].append(validation_loss)
        
        for metric_name, metric_value in validation_metrics.items():
            training_history[metric_name].append(metric_value)
        
        #  [LOGGING]: Mostrar progreso detallado de la época
        print(
            f" Época {epoch:2d}/{EPOCHS_FINAL} → "
            f"Train Loss: {train_loss:.4f} | Val Loss: {validation_loss:.4f}"
        )
        print(
            f"     ROC: {validation_metrics['roc']:.4f} | "
            f"PR: {validation_metrics['pr']:.4f} | "
            f"Brier: {validation_metrics['brier']:.4f}"
        )
        print(
            f"     Acc: {validation_metrics['accuracy']:.4f} | "
            f"Prec: {validation_metrics['precision']:.4f} | "
            f"Rec: {validation_metrics['recall']:.4f} | "
            f"F1: {validation_metrics['f1']:.4f} | "
            f"F2: {validation_metrics['f2']:.4f}"
        )
        
        # 
        # ⏰ EARLY STOPPING BASADO EN F2-SCORE
        # 
        
        current_f2 = validation_metrics['f2']
        
        if current_f2 > best_f2_final:
            #  [MEJOR MODELO]: Nuevo mejor F2-score encontrado
            best_f2_final = current_f2
            patience_counter = 0
            
            #  [CHECKPOINT]: Guardar estado del mejor modelo
            torch.save(final_model.state_dict(), REPORT_DIR / "best_model.pth")
            print(f"     Nuevo mejor F2: {best_f2_final:.4f} - Modelo guardado")
            
        else:
            # ⏰ [PACIENCIA]: Incrementar contador de épocas sin mejora
            patience_counter += 1
            print(f"    ⏳ Sin mejora: {patience_counter}/{PATIENCE} épocas")
            
            #  [PARADA TEMPRANA]: Detener si se agotó la paciencia
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping activado: {PATIENCE} épocas sin mejora en F2-score")
                print(f" Mejor F2-score alcanzado: {best_f2_final:.4f}")
                break
        
        print()  # Línea en blanco para separar épocas
    
    show_gpu(" Después del entrenamiento final")
    
    # 
    #  PERSISTENCIA DEL HISTÓRICO DE ENTRENAMIENTO
    # 
    
    #  [HISTÓRICO]: Guardar evolución completa de métricas
    with open(REPORT_DIR / "history.json", "w") as f:
        json.dump(training_history, f, indent=2)
    
    print(f" Histórico de entrenamiento guardado en {REPORT_DIR / 'history.json'}")
    
    # 
    #  EVALUACIÓN FINAL EN CONJUNTO DE PRUEBA
    # 
    
    print(f"\n Evaluación final en conjunto de prueba...")
    
    #  [MEJOR MODELO]: Cargar pesos del mejor modelo según validación
    final_model.load_state_dict(torch.load(REPORT_DIR / "best_model.pth"))
    
    #  [EVALUACIÓN FINAL]: Obtener métricas en conjunto de prueba
    y_true_test, y_prob_test = evaluate_probs(final_model, test_loader)
    final_test_metrics = evaluate_metrics(y_true_test, y_prob_test)
    
    #  [MÉTRICAS FINALES]: Guardar resultados en conjunto de prueba
    with open(REPORT_DIR / "test_metrics.json", "w") as f:
        json.dump(final_test_metrics, f, indent=2)
    
    #  [REPORTE FINAL]: Mostrar métricas finales formateadas
    print(f"\n MÉTRICAS FINALES EN CONJUNTO DE PRUEBA:")
    print(f"")
    print(f" ROC-AUC Score:      {final_test_metrics['roc']:.4f}")
    print(f" PR-AUC Score:       {final_test_metrics['pr']:.4f}")
    print(f" Brier Score:        {final_test_metrics['brier']:.4f}")
    print(f" Accuracy:           {final_test_metrics['accuracy']:.4f}")
    print(f" Precision:          {final_test_metrics['precision']:.4f}")
    print(f" Recall:             {final_test_metrics['recall']:.4f}")
    print(f" F1-Score:           {final_test_metrics['f1']:.4f}")
    print(f" F2-Score:           {final_test_metrics['f2']:.4f}")
    
    #  [RESUMEN]: Mostrar ubicación de todos los artefactos generados
    print(f"\n ARTEFACTOS GENERADOS EN {REPORT_DIR}:")
    print(f"    best_model.pth     → Pesos del mejor modelo")
    print(f"    history.json       → Histórico de entrenamiento")
    print(f"    optuna_study.pkl   → Estudio de optimización HPO")
    print(f"    test_metrics.json  → Métricas finales de prueba")
    
    print(f"\n Pipeline de entrenamiento completado exitosamente!")
    
    show_gpu(" Final del pipeline")


# 
#  PUNTO DE ENTRADA PRINCIPAL
# 

if __name__ == "__main__":
    print(" Iniciando Sistema de Evaluación de Riesgo Crediticio")
    print("=" * 60)
    main()
    print("=" * 60)
    print(" Sistema completado exitosamente")