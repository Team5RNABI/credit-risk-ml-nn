# Credit Risk ML NN

Proyecto de modelado de riesgo crediticio para clasificar clientes segun su probabilidad de incumplimiento, comparando un enfoque de scorecard tradicional con una red neuronal hibrida con embeddings para variables categoricas.

## Resumen

El objetivo del proyecto es construir un pipeline reproducible para:

- preparar datos de riesgo crediticio;
- entrenar y evaluar modelos de clasificacion binaria;
- comparar un modelo champion basado en scorecard/logistic regression con un challenger de deep learning;
- generar artefactos de entrenamiento, metricas y material de analisis.

El enfoque principal combina:

- variables numericas normalizadas;
- embeddings para variables categoricas;
- optimizacion de hiperparametros con Optuna;
- entrenamiento con PyTorch;
- evaluacion con metricas orientadas a clases desbalanceadas.

## Caracteristicas

- Pipeline completo de entrenamiento en `train.py`.
- Modelado de riesgo crediticio con red neuronal hibrida.
- Comparacion champion vs challenger.
- Soporte para experimentacion con notebooks.
- Generacion de metricas, historial y artefactos de modelo en `reports/`.
- Documentacion narrativa para notebooks.

## Estructura del repositorio

- `app/`: espacio para una aplicacion o capa de inferencia.
- `data/`: datos de trabajo y material de entrada.
- `notebooks/`: notebooks de exploracion, modelado y evaluacion.
- `reports/`: artefactos generados por entrenamiento y analisis.
- `train.py`: script principal de entrenamiento.
- `requirements.txt`: dependencias del proyecto.
- `prompt_documentation.md`: guia para documentacion del proyecto.
- `prompt_documentation_notebook.md`: guia para documentacion de notebooks.

## Flujo de trabajo

1. Preparar los datos en el formato esperado.
2. Ejecutar el preprocesamiento y la exploracion.
3. Entrenar el modelo challenger con `train.py`.
4. Comparar metricas y artefactos generados.
5. Revisar notebooks y reportes para analisis final.

## Requisitos

- Python 3.11 o superior.
- Las librerias indicadas en `requirements.txt`.
- Un entorno virtual recomendado para aislar dependencias.

## Instalacion

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Preparacion de datos

El script de entrenamiento espera un dataset procesado en:

```text
data/processed/data_loan_complete.csv
```

Ese archivo debe contener las variables numericas, categoricas y la columna objetivo usada por el pipeline.

## Entrenamiento

Ejecuta el pipeline principal con:

```bash
python train.py
```

Durante la ejecucion se generan artefactos en `reports/`, incluyendo:

- pesos del mejor modelo;
- historial de entrenamiento;
- resultados de Optuna;
- metricas finales;
- configuracion resumida del modelo.

## Notebooks

La carpeta `notebooks/` contiene el recorrido analitico del proyecto:

- `01_preprocessing.ipynb`: carga y preparacion de datos;
- `02_eda.ipynb`: analisis exploratorio;
- `03_nn_model.ipynb`: desarrollo y evaluacion del modelo neuronal;
- `04_scorecard_scaling.ipynb`: comparacion y escalado del scorecard;
- `05_evaluation.ipynb`: espacio para validacion final y analisis complementario.

## Artefactos generados

El proyecto produce salidas utiles para seguimiento y reproduccion:

- `reports/history.json`
- `reports/metrics.json`
- `reports/test_metrics.json`
- `reports/optuna_study.pkl`
- `reports/best_model.pth`
- artefactos de comparacion y analisis en `reports/modeloFinal/`

## Reproducibilidad

El pipeline usa semilla fija y registra configuraciones clave para facilitar experimentos comparables. Si cambias datos, hiperparametros o estructura de features, conviene volver a ejecutar el entrenamiento desde cero.

## Recomendaciones de uso

- Mantener una copia limpia del dataset de entrada.
- Ejecutar el entrenamiento desde un entorno virtual.
- Revisar los notebooks antes de publicar resultados.
- Conservar los artefactos generados solo para reproduccion local o despliegue interno.

