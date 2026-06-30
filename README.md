# Credit Risk ML NN

Proyecto academico y tecnico para modelado de riesgo crediticio orientado a estimar la probabilidad de incumplimiento de clientes a partir de variables financieras y de comportamiento. El repositorio compara un enfoque de scorecard tradicional con un challenger basado en una red neuronal hibrida con embeddings para variables categoricas.

## Resumen ejecutivo

Este proyecto implementa un pipeline reproducible para trabajar con datos de riesgo crediticio y comparar dos enfoques de modelado:

- un modelo champion de corte tradicional, basado en scorecard y regresion logistica;
- un modelo challenger de deep learning, entrenado con variables numericas y embeddings para variables categoricas.

El objetivo es disponer de una base ordenada para preparacion de datos, entrenamiento, evaluacion, trazabilidad de experimentos y generacion de artefactos listos para revision tecnica o academica.

## Problema y alcance

El problema central consiste en estimar la probabilidad de default de un cliente a partir de su informacion historica y financiera. Esta estimacion permite apoyar decisiones de aprobacion, ajuste de politicas de riesgo y comparacion entre alternativas de modelado.

Desde una perspectiva academica, el proyecto permite estudiar:

- tratamiento de datos tabulares desbalanceados;
- ingenieria de variables numericas y categoricas;
- comparacion entre modelos interpretables y modelos no lineales;
- impacto de la calibracion y de metricas orientadas a recall en contextos financieros.

## Enfoque metodologico

El flujo de trabajo sigue una logica de experimentacion controlada:

1. preparacion y validacion del dataset;
2. normalizacion de variables numericas;
3. codificacion de variables categoricas mediante embeddings;
4. entrenamiento con PyTorch;
5. busqueda de hiperparametros con Optuna;
6. evaluacion con metricas robustas para clases desbalanceadas;
7. persistencia de resultados y artefactos.

Arquitectura conceptual:

```text
Datos procesados
    -> Preparacion / limpieza
    -> Split train / valid / test
    -> Baseline scorecard
    -> Challenger neuronal
    -> Optimizacion y early stopping
    -> Evaluacion final
    -> Reports / artifacts
```

Esta combinacion permite capturar relaciones no lineales en datos tabulares sin perder orden ni trazabilidad en el proceso experimental.

## Estrategia de ramas

La rama `main` se utiliza como version estable y consolidada del proyecto. Esta decision es util por tres razones:

- mantiene un punto de referencia limpio para revision;
- separa la experimentacion del resultado final;
- reduce el riesgo de mezclar codigo exploratorio con entregables listos para analisis.

En terminos practicos, la organizacion por ramas ayuda a trabajar con iteraciones de modelado o documentacion sin comprometer la version principal. La rama `main` debe leerse como la presentacion curada del proyecto.

## Objetivos

### Objetivo principal

- construir y evaluar un pipeline de riesgo crediticio que compare un baseline tradicional frente a un modelo neuronal hibrido.

### Objetivos secundarios

- documentar el flujo de preparacion, modelado y evaluacion;
- generar artefactos reproducibles del entrenamiento;
- conservar una narrativa tecnica clara para auditoria y analisis;
- facilitar la extension del proyecto hacia inferencia o despliegue posterior.

## Criterios de evaluacion

El proyecto utiliza metricas adecuadas para problemas binarios desbalanceados:

- ROC-AUC: mide capacidad discriminativa general.
- PR-AUC: enfatiza desempeno en clase positiva rara.
- Brier Score: evalua calibracion de probabilidades.
- F2-Score: prioriza recall sobre precision, util en escenarios donde perder positivos es costoso.

## Contrato de datos

El pipeline de entrenamiento espera un archivo procesado en:

```text
data/processed/data_loan_complete.csv
```

El modelo consolidado en esta version trabaja con las siguientes variables:

### Variables numericas

- `loan_amnt`
- `annual_inc`
- `dti`
- `revol_util`
- `total_acc`
- `tot_cur_bal`
- `days_since_earliest_cr`

### Variables categoricas

- `emp_length`
- `home_ownership`
- `purpose`
- `delinq_2yrs`
- `inq_last_6mths`
- `pub_rec`

La columna objetivo esperada es `target`.

## Estructura del repositorio

| Ruta | Proposito |
| --- | --- |
| `app/` | Base para una futura capa de inferencia o servicio |
| `data/` | Datos de trabajo y material de entrada |
| `notebooks/` | Notebooks de preparacion, EDA, modelado y evaluacion |
| `reports/` | Artefactos generados por entrenamiento y analisis |
| `train.py` | Script principal de entrenamiento |
| `requirements.txt` | Dependencias del proyecto |
| `prompt_documentation.md` | Guia de documentacion general |
| `prompt_documentation_notebook.md` | Guia de documentacion para notebooks |
| `LICENSE` | Licencia del proyecto |

## Requisitos

- Python 3.11 o superior.
- Un entorno virtual recomendado.
- Dependencias instaladas desde `requirements.txt`.

## Instalacion

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Si quieres validar primero el entorno, revisa `requirements.txt` y ajusta versiones segun tu sistema.

## Uso

### 1. Preparar datos

Asegurate de contar con el dataset procesado en:

```text
data/processed/data_loan_complete.csv
```

### 2. Ejecutar entrenamiento

```bash
python train.py
```

### 3. Revisar resultados

Al finalizar, revisa los artefactos generados en `reports/` para validar metricas, historial de entrenamiento y configuracion del mejor modelo.

### 4. Seguir el recorrido analitico

Los notebooks estan organizados en orden secuencial:

1. `notebooks/01_preprocessing.ipynb`
2. `notebooks/02_eda.ipynb`
3. `notebooks/03_nn_model.ipynb`
4. `notebooks/04_scorecard_scaling.ipynb`
5. `notebooks/05_evaluation.ipynb`

## Pipeline tecnico

### Preparacion

- carga de datos;
- limpieza inicial;
- validacion de tipos y consistencia;
- separacion de variables numericas y categoricas.

### Modelado

- construccion del dataset para PyTorch;
- entrenamiento de la red neuronal hibrida;
- busqueda de hiperparametros con Optuna;
- early stopping basado en F2.

### Evaluacion

- metricas en validacion y prueba;
- comparacion contra el baseline;
- revision de calibracion y capacidad discriminativa;
- persistencia de resultados.

## Artefactos generados

La ejecucion del pipeline produce resultados como:

- `reports/best_model.pth`
- `reports/optuna_study.pkl`
- `reports/history.json`
- `reports/metrics.json`
- `reports/test_metrics.json`
- `reports/modeloFinal/`

Estos archivos permiten comparar corridas, conservar configuraciones y reutilizar el mejor estado del modelo.

## Reproducibilidad

El flujo trabaja con semilla fija y guarda configuraciones clave para facilitar comparaciones entre experimentos. Si cambian los datos, el esquema de variables o los hiperparametros, conviene ejecutar nuevamente el pipeline completo.

## Documentacion del proyecto

Ademas del codigo, el repositorio incluye una narrativa tecnica para acompanar el analisis:

- `prompt_documentation.md`: guia general de estilo y estructura del proyecto.
- `prompt_documentation_notebook.md`: guia para construir notebooks con narrativa academica y explicativa.

## Recomendaciones de uso

- Mantener una copia limpia del dataset de entrada.
- Ejecutar los notebooks en orden para preservar el flujo analitico.
- Revisar `reports/` antes de reutilizar resultados en presentaciones o validaciones.
- Trabajar siempre desde un entorno virtual.

## Licencia

Este proyecto usa licencia MIT. Revisa `LICENSE` para ver los terminos completos.
