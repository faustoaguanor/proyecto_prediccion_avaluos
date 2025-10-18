# Proyecto de Predicción de Avalúos Catastrales con Detección de Target Leakage

## Descripción

Sistema completo de Machine Learning para predicción de avalúos catastrales con:
- **Detección automática de target leakage** (fuga de información)
- **Dos experimentos paralelos** para cuantificar el impacto de features sospechosas
- Pipeline completo: limpieza, EDA, feature engineering, modelado y evaluación
- Modelos de regresión, clasificación y clustering

## Requisitos

```bash
pip install -r requirements.txt
```

## Estructura del Proyecto

```
catastro_prediccion/
├── src/                    # Módulos del pipeline
├── notebooks/              # Notebook de exploración
├── output/                 # Resultados y reportes
├── main.py                 # Script principal
└── predio_10.csv          # Datos de entrada
```

## Uso

### Ejecución completa del pipeline:

```bash
python main.py
```

Esto generará:
- `output/leakage_report.json`: Reporte de detección de leakage
- `output/summary.html`: Reporte completo con métricas y visualizaciones
- `output/models/`: Modelos entrenados guardados
- `output/figures/`: Gráficas generadas

### Exploración interactiva:

```bash
jupyter notebook notebooks/exploracion_entrenamiento.ipynb
```

## Características Principales

### 1. Detección de Target Leakage

El sistema detecta automáticamente columnas que pueden filtrar información de la variable objetivo:

- **Por nombre**: Columnas con patrones como `valor`, `valoracion`, `avaluo`, `precio`, etc.
- **Por estadística**: Correlación > 0.8 o feature importance muy alta

### 2. Experimentos Paralelos

- **Experimento A** (sin fuga): Excluye features sospechosas
- **Experimento B** (con fuga): Incluye todas las features para cuantificar el impacto

### 3. Modelos Implementados

**Regresión:**
- Linear Regression, Ridge, Lasso, ElasticNet
- Random Forest, Gradient Boosting
- XGBoost, LightGBM

**Clasificación (por rangos):**
- Logistic Regression, Random Forest
- XGBoost, KNN

**Clustering:**
- KMeans, DBSCAN, Agglomerative

### 4. Validación y Optimización

- Cross-validation (k=5)
- GridSearch/RandomizedSearch para hiperparámetros
- Métricas: R², MAE, RMSE, MAPE

## Formato del CSV

- **Separador**: `;` (punto y coma)
- **Decimales**: `,` (coma) - ejemplo: `1.234,56`
- **Encoding**: UTF-8 o latin1 (detección automática)

## Salida del Reporte

El `leakage_report.json` contiene:
- Columnas detectadas como sospechosas
- Métricas comparativas (Exp A vs Exp B)
- Recomendaciones sobre qué features usar en producción

El `summary.html` incluye:
- Análisis exploratorio con gráficas
- Comparación de modelos
- Tabla de importancia de features
- Visualización del impacto del leakage

## Autor

Proyecto profesional para análisis catastral con enfoque en prevención de data leakage.
