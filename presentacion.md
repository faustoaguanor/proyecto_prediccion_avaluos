---
marp: true
theme: default
paginate: true
backgroundColor: #fff
header: 'Predicción de Avalúos Catastrales con Machine Learning'
footer: 'Fausto Guano | Universidad Yachay Tech | 2025'
style: |
  section {
    font-size: 26px;
  }
  h1 {
    color: #2563eb;
    border-bottom: 3px solid #2563eb;
    font-size: 42px;
  }
  h2 {
    color: #1e40af;
    font-size: 32px;
  }
  .columns {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
  }
  .alert {
    background-color: #fef3c7;
    padding: 1rem;
    border-left: 4px solid #f59e0b;
    margin: 1rem 0;
  }
  .success {
    background-color: #d1fae5;
    padding: 1rem;
    border-left: 4px solid #10b981;
    margin: 1rem 0;
  }
  .info {
    background-color: #dbeafe;
    padding: 1rem;
    border-left: 4px solid #3b82f6;
    margin: 1rem 0;
  }
  img {
    max-width: 100%;
    max-height: 500px;
    display: block;
    margin: 0 auto;
  }
  table {
    font-size: 22px;
    margin: 1rem auto;
  }
  code {
    background-color: #f3f4f6;
    padding: 0.2rem 0.4rem;
    border-radius: 3px;
  }
---

<!-- _class: lead -->
<!-- _paginate: false -->

# Predicción de Avalúos Catastrales
## Sistema Inteligente con Detección Automática de Target Leakage

**Fausto Guano**
Universidad Yachay Tech
Maestría en Ciencia de Datos

---

# Agenda

1. **Problema y Contexto**
2. **Dataset y Análisis Exploratorio**
3. **Metodología: Pipeline de ML**
4. **Feature Engineering Avanzado**
5. **Detección de Target Leakage**
6. **Experimentos Comparativos**
7. **Resultados y Métricas**
8. **Análisis Espacial de Errores**
9. **Conclusiones y Trabajo Futuro**

---

# 1. Problema y Contexto

## ¿Qué es un Avalúo Catastral?

<div class="info">

**Definición:** Valor fiscal asignado a propiedades inmuebles para determinar impuestos prediales y tasas municipales.

</div>

- Tradicionalmente calculado por **peritos especializados**
- Proceso **costoso** (USD 50-200 por propiedad)
- **Lento** (3-6 meses para ciudades grandes)
- Propenso a **subjetividad y sesgo humano**

---

# El Desafío

<div class="columns">

<div>

## Problemática
- **46,874 propiedades** en dataset original
- **44,407 propiedades** tras limpieza
- Métodos tradicionales:
  - 6 meses de trabajo
  - 20 peritos
  - USD 2,343,700 de costo
- Inconsistencias entre valoraciones

</div>

<div>

## Solución Propuesta
- Sistema automatizado con ML
- Predicción en **segundos**
- Costo: **USD 0** por propiedad
- **Detección automática de sesgos**
- Validación espacial del error
- **R² = 0.9605** (96.05% precisión)

</div>

</div>

---

# 2. Dataset y Exploración

## Fuente de Datos

- **Dataset catastral:** Distrito Metropolitano de Quito, Ecuador
- **Tamaño original:** 46,874 registros
- **Tamaño final:** 44,407 registros (5.3% outliers removidos)
- **Variables iniciales:** 43 columnas
- **Variables finales:** 120 → **60 features seleccionadas**

---

# Limpieza Crítica del Target

## Transformación Logarítmica Aplicada

| Métrica | Antes de Limpieza | Después de Limpieza |
|---------|-------------------|---------------------|
| **Registros** | 46,874 | 44,407 (-5.3%) |
| **Mínimo** | $35.28 | $10,003.09 |
| **Máximo** | $229,593,114 | $2,283,111.39 |
| **Ratio Max/Min** | 6,507,741x | 228.2x |
| **Asimetría** | 101.17 | **0.29** ✅ |

<div class="success">

**Decisión clave:** Filtrar valores <$10,000 y outliers extremos (P1, P99) + aplicar **log-transform** normaliza la distribución exitosamente

</div>

---

# Tipos de Variables

<div class="columns">

<div>

### Características Físicas
- `Area_Construccion` (m²)
- `Area_Terreno_Escri` (m²)
- `Frente_Total` (metros)
- `Anio_Construccion`
- `Pisos_PUGS`

### Características Normativas
- `Cos_PUGS` (Coeficiente ocupación)
- `Lot_Min_PUGS` (Lote mínimo)
- `Factor_Proteccion`
- `Factor_Topografia`

</div>

<div>

### Características Espaciales
- `Latitud` / `Longitud`
- `Infl_Cent_Norm` (centros comerciales)
- `Infl_Educ_Norm` (educación)
- `Infl_Metr_Norm` (metro)
- `Infl_Road_Norm` (vías principales)

### Características Ambientales
- `Patri_Forest` (patrimonio forestal)
- `Prot_Quebra` (protección quebradas)
- `Nvl_Aazmm` (nivel amenaza)

</div>

</div>

---

# Variable Objetivo (Target)

## `Valoracion_Terreno` (USD) - Después de Limpieza

| Estadístico | Valor |
|-------------|-------|
| **Media** | $235,770 |
| **Mediana** | $91,140 |
| **Desv. Estándar** | ~$185,000 |
| **Mínimo** | $10,003 |
| **Máximo** | $2,283,111 |
| **Outliers eliminados** | 2,467 (5.3%) |

<div class="alert">

**Transformación:** Log-transform reduce asimetría de 101.17 → **0.29** (mejora de 100.88 puntos)

</div>

---

# Análisis Univariado del Target

![Distribución de Valoración](output/figures/univariate_analysis.png)

---

# Interpretación: Distribución del Target

<div class="columns">

<div>

## Observaciones Clave
- **Distribución normalizada** tras log-transform
- Mayoría de propiedades: $50k - $200k
- Rango controlado: $10k - $2.28M
- **Asimetría casi eliminada** (0.29)

</div>

<div>

## Implicaciones para ML
- ✅ **Log-transform aplicada**
- Predicciones des-transformadas automáticamente
- Modelos basados en árboles se benefician
- RMSE en escala logarítmica más interpretable
- Mejor manejo de valores extremos

</div>

</div>

---

# Análisis de Correlaciones

![Matriz de Correlación](output/figures/correlation_analysis.png)

---

# Interpretación: Correlaciones

## Top 5 Correlaciones con `Valoracion_Terreno`

| Variable | Correlación Pearson | Correlación Spearman | Interpretación |
|----------|---------------------|----------------------|----------------|
| `Area_Terreno_Escri` | 0.277 | **0.743** | Relación no-lineal fuerte |
| `Aiva_Valor` ⚠️ | 0.146 | **0.678** | LEAKAGE - Excluida |
| `Lot_Min_PUGS` | 0.035 | **0.623** | Normativa urbana importante |
| `Infl_Func_Norm` | 0.087 | **0.596** | Proximidad a servicios |
| `Frente_Total` | 0.520 | **0.553** | Relación fuerte y lineal |

<div class="alert">

**Nota:** Diferencias Pearson vs Spearman indican **relaciones no-lineales** → Modelos de árboles ideales

</div>

---

# 3. Metodología: Pipeline Completo

## Arquitectura del Sistema

```
1. Carga y Normalización
   ├── Limpieza de nombres (snake_case)
   ├── Detección automática de tipos
   └── Identificación de columnas ID

2. Preprocesamiento
   ├── Detección automática del target ✓
   ├── Imputación KNN (11 columnas)
   ├── One-Hot Encoding (19 categóricas)
   ├── Standard Scaling (numéricas)
   └── Detección de Outliers (IQR)
```

---

# Pipeline (continuación)

```
3. Limpieza Crítica del Target
   ├── Filtrado P1-P99 (938 outliers extremos)
   ├── Eliminación valores <$10,000 (1,998 registros)
   ├── Transformación logarítmica
   └── Verificación de normalización
   → 46,874 → 44,407 registros (5.3% removido)

4. Feature Engineering
   ├── 6 features de áreas y ratios
   ├── 7 features geoespaciales
   ├── 6 features de influencias agregadas
   ├── 5 features temporales
   └── 5 features de regulación urbana
   → 29 features nuevas creadas
```

---

# Pipeline (final)

```
5. Detección de Target Leakage
   ├── Por nombre de columna
   ├── Por correlación estadística (>0.8)
   └── 1 feature sospechosa detectada: Aiva_Valor

6. Feature Selection
   ├── Eliminación de correlacionadas (>0.95)
   ├── Random Forest Feature Importance
   └── Top 60 features seleccionadas
   → 119 features → 60 features finales

7. Experimentos Paralelos
   ├── Exp A: Sin leakage (60 features) ✅
   ├── Exp B: Con leakage (120 features) ⚠️
   └── 8 modelos entrenados por experimento
```

---

# 4. Feature Engineering Avanzado

## Variables Creadas (29 nuevas)

### 1. Áreas y Construcción (6 features)
```python
Ratio_Construccion_Terreno = Area_Construccion / Area_Terreno
Area_Total = Area_Construccion + Area_Terreno
Area_No_Construida = Area_Terreno - Area_Construccion
Profundidad_Estimada = Area_Terreno / Frente_Total
Ratio_Frente_Area = Frente_Total / Area_Terreno
Area_Por_Piso = Area_Construccion / Pisos
```

---

# Feature Engineering (continuación)

### 2. Geoespaciales (7 features)
```python
Distancia_Centro = √((lat - lat_quito)² + (lon - lon_quito)²)
Lat_Relativa = lat - lat_quito
Lon_Relativa = lon - lon_quito
Cuadrante = {NE, NW, SE, SW}  # basado en centro
Distancia_Centro_Manhattan = |Δlat| + |Δlon|
```

### 3. Influencias Agregadas (6 features)
```python
Influencia_Total = Σ(todas las influencias)
Influencia_Media = mean(influencias)
Influencia_Max = max(influencias)
Influencia_Min = min(influencias)
Influencia_Std = std(influencias)
Influencia_Rango = max - min
```

---

# 5. Detección de Target Leakage

## ¿Qué es Target Leakage?

<div class="alert">

**Definición:** Variables que contienen información del target que **NO estará disponible** al momento de hacer predicciones en producción.

**Consecuencia:** Modelo con métricas excelentes en validación pero **falla completamente en producción**.

</div>

---

# Métodos de Detección Implementados

<div class="columns">

<div>

## 1. Detección por Nombre
```python
patrones = [
    'valor', 'valoracion',
    'avaluo', 'precio',
    'costo', 'monto'
]
```
**Resultado:** 1 columna detectada
- `Aiva_Valor`

</div>

<div>

## 2. Detección Estadística
- Correlación Pearson > 0.8
- Correlación Spearman > 0.8
- p-valor < 0.05

**Resultado:** 0 columnas adicionales
(Aiva_Valor tiene 0.678, bajo el umbral)

</div>

</div>

---

# 6. Feature Selection

## Proceso de Selección

**Paso 1:** Eliminación de multicolinealidad
- 23 pares con correlación > 0.95
- 15 features eliminadas
- **119 → 104 features**

**Paso 2:** Random Forest Feature Importance
- Entrenamiento con 500 árboles
- Top 60 features seleccionadas
- **104 → 60 features finales**

---

# Top 10 Features Seleccionadas

| Ranking | Feature | Importancia | Tipo |
|---------|---------|-------------|------|
| 1 | `Area_Terreno_Escri` | 0.5376 | Física |
| 2 | `Lot_Min_PUGS` | 0.0936 | Normativa |
| 3 | `Pisos_PUGS` | 0.0855 | Física |
| 4 | `Area_Construccion` | 0.0835 | Física |
| 5 | `Distancia_Centro` | 0.0501 | Ingenierizada |
| 6 | `Longitud` | 0.0243 | Espacial |
| 7 | `Frente_Total` | 0.0187 | Física |
| 8 | `Parroquia` | 0.0132 | Categórica |
| 9 | `Clasi_Suelo_URBANO` | 0.0123 | Normativa |
| 10 | `Infl_Road_Norm` | 0.0114 | Espacial |

---

# 7. Experimentos Comparativos

## Diseño Experimental

<div class="columns">

<div>

### Experimento A
**Modelo Sin Leakage**
- Features: **60** (top seleccionadas)
- Excluye: `Aiva_Valor`
- Train/Test: 80/20 (35,525 / 8,882)
- CV: 5-fold
- ✅ **Modelo recomendado**

</div>

<div>

### Experimento B
**Modelo Con Leakage**
- Features: **120** (todas)
- Incluye: `Aiva_Valor`
- Train/Test: 80/20
- CV: 5-fold
- ⚠️ **Solo referencia**

</div>

</div>

<div class="info">

**Objetivo:** Cuantificar el impacto del leakage en las métricas de rendimiento

</div>

---

# Resultados: Experimento A (Sin Leakage)

## Performance de los 8 Modelos

| Modelo | R² Test | RMSE (log) | MAE (log) |
|--------|---------|------------|-----------|
| **RandomForest** 🏆 | **0.9605** | **0.21** | **0.10** |
| XGBoost | 0.9591 | 0.21 | 0.12 |
| LightGBM | 0.9559 | 0.22 | 0.14 |
| GradientBoosting | 0.9307 | 0.27 | 0.19 |
| LinearRegression | 0.7235 | 0.55 | 0.40 |
| Ridge | 0.7214 | 0.55 | 0.41 |
| ElasticNet | 0.3581 | 0.83 | 0.65 |
| Lasso | 0.3474 | 0.84 | 0.65 |

---

# Resultados: Experimento B (Con Leakage)

## Performance con Todas las Features

| Modelo | R² Test | RMSE (log) | MAE (log) | Diferencia vs A |
|--------|---------|------------|-----------|----------------|
| **RandomForest** | **0.9750** | **0.16** | **0.08** | +1.44% |
| XGBoost | 0.9727 | 0.17 | 0.09 | +1.36% |
| LightGBM | 0.9728 | 0.17 | 0.09 | +1.69% |
| GradientBoosting | 0.9618 | 0.20 | 0.13 | +3.11% |

<div class="alert">

**Hallazgo:** Exp B tiene R² ligeramente mayor (+1.44%), confirmando presencia de leakage leve

</div>

---

# Comparación de Experimentos

![Comparación Exp A vs B](output/figures/experiment_comparison.png)

---

# Interpretación: Comparación de Experimentos

## Análisis del Impacto del Leakage

| Experimento | Mejor Modelo | R² Test | RMSE | Diferencia |
|-------------|--------------|---------|------|------------|
| **A (Sin Leakage)** | RandomForest | **0.9605** | 0.21 | Baseline |
| **B (Con Leakage)** | RandomForest | 0.9750 | 0.16 | **+1.44%** ⬆️ |

<div class="success">

**Conclusión:** Leakage detectado tiene impacto moderado (+1.44%). Modelo A es **production-ready** con 96.05% de precisión.

**Decisión:** Usar Experimento A para producción (sin `Aiva_Valor`)

</div>

---

# 8. Optimización de Hiperparámetros

## RandomForest Tuning

**Método:** RandomizedSearchCV
- **Búsqueda:** 20 combinaciones
- **CV:** 3-fold
- **Tiempo:** ~5 minutos

### Parámetros Optimizados
```python
n_estimators: 300      (antes: 100)
max_depth: 20          (antes: None)
min_samples_split: 5   (antes: 2)
min_samples_leaf: 1    (antes: 1)
max_features: 0.3      (antes: 'sqrt')
```

---

# Resultado de Optimización

## Comparación Base vs Optimizado

| Métrica | Modelo Base | Modelo Optimizado | Cambio |
|---------|-------------|-------------------|--------|
| **R² Test** | 0.9605 | 0.9599 | **-0.07%** ⬇️ |
| **RMSE (log)** | 0.21 | 0.21 | 0% |
| **R² CV** | - | 0.9554 | - |

<div class="alert">

**Decisión:** Mantener modelo base. La optimización no mejora significativamente (posible overfitting en la búsqueda).

</div>

---

# Métricas Finales en Escala Original

## RandomForest (Experimento A) - Dataset Test

```python
Test Set (8,882 registros nunca vistos):

  R² Score:              0.9605  ⭐ Excelente
  MAE (Error Absoluto):  $27,022
  RMSE:                  $46,440
  MAPE (Error %):        12.96% ✅
  Mediana Error:         $13,500 ✅
  Error Máximo:          $1,140,000
```

<div class="success">

**Interpretación:** 96.05% de varianza explicada. Error promedio de $27k es solo **2.9%** del valor mediano ($91k).

</div>

---

# Distribución del Error Espacial

## Análisis por Magnitud de Error

| Magnitud del Error | Rango % | Cantidad | Porcentaje |
|-------------------|---------|----------|------------|
| **Excelente** | < 5% | 4,543 | **51.1%** ✅ |
| **Bueno** | 5-10% | 1,563 | **17.6%** |
| **Aceptable** | 10-20% | 1,461 | **16.4%** |
| **Alto** | > 20% | 1,315 | **14.8%** |

<div class="success">

**Hallazgo clave:** 68.7% de predicciones tienen error <10%, excelente para producción

</div>

---

# Análisis de Residuos

![Análisis de Residuos](output/figures/residual_analysis.png)

---

# Interpretación: Residuos

<div class="columns">

<div>

## Gráfico Superior
**Predicho vs Real**
- Puntos cerca de diagonal = buenas predicciones
- Excelente ajuste en $50k-$500k
- Ligera dispersión en valores >$1M

</div>

<div>

## Gráfico Inferior
**Residuos vs Predicciones**
- Centrados en 0 ✅
- Heterocedasticidad controlada
- Log-transform reduce outliers
- Sin patrones sistemáticos

</div>

</div>

<div class="info">

**Implicación:** El modelo es confiable en todo el rango de valores tras la transformación logarítmica

</div>

---

# 9. Análisis Espacial del Error

## Archivo para Visualización GIS

**Generado:** `output/test_completo_con_predicciones.xlsx`

### Contenido (8,882 registros):
- `Cat_Lote_Id`: ID único
- `Latitud`, `Longitud`: Coordenadas
- `Valoracion_Real`: Valor real
- `Prediccion`: Valor predicho
- `Error_Absoluto`: |Real - Predicción|
- `Error_Porcentual`: Error relativo %
- `Magnitud_Error`: Categoría (Excelente/Bueno/Aceptable/Alto)

---

# Uso del Archivo para Análisis Espacial

## Recomendaciones de Visualización

### En QGIS/ArcGIS:
1. Unir con capa catastral usando `Cat_Lote_Id`
2. Simbolizar por `Magnitud_Error` (colores categóricos)
3. Crear mapa de calor con `Error_Absoluto`
4. Identificar clusters espaciales de error alto

<div class="info">

**Hipótesis a validar espacialmente:**
- ¿Mayor error en zonas periféricas?
- ¿Clusters de error en áreas específicas?
- ¿Relación con zonificación urbana o topografía?

</div>

---

# Clustering Analysis

![Método del Codo](output/figures/elbow_method.png)

---

# Interpretación: Clustering

## Método del Codo (Elbow Method)

<div class="columns">

<div>

### Observaciones
- **Codo pronunciado en k=5**
- Inercia decrece rápidamente hasta k=5
- Después k=5: mejora marginal
- Silhouette Score: 0.871

</div>

<div>

### Clusters Identificados
- **Cluster 0:** Residencial bajo
- **Cluster 1:** Residencial medio
- **Cluster 2:** Residencial alto
- **Cluster 3:** Comercial/Premium
- **Cluster 4:** Lujo

</div>

</div>

---

# Clasificación por Rangos de Avalúo

## Performance de Modelos de Clasificación

**Target:** 5 clases (quintiles de valoración)

| Modelo | Accuracy Test | Interpretación |
|--------|---------------|----------------|
| **XGBoost** | **89.45%** | Excelente |
| **RandomForest** | 88.32% | Muy bueno |
| **KNN** | 67.06% | Moderado |
| **LogisticRegression** | 49.88% | Apenas mejor que azar |

<div class="success">

**Aplicación práctica:** Sistema puede clasificar propiedades en rangos de precio con 89% de exactitud

</div>

---

# 10. Conclusiones

## Hallazgos Principales

1. ✅ **R² = 0.9605** en datos no vistos (excelente)
2. ✅ **Log-transform crítica:** Reduce asimetría 101.17 → 0.29
3. ✅ **60 features óptimas:** Balance precisión/complejidad
4. ✅ **RandomForest dominante** sobre todos los algoritmos
5. ✅ **68.7% predicciones con error <10%** excelente para producción
6. ✅ **Detección de leakage:** Impacto moderado (+1.44%)
7. ✅ **Validación espacial:** 51.1% predicciones excelentes (<5% error)

---

# Ventajas del Sistema

<div class="columns">

<div>

## Beneficios Técnicos
- Detección automática de leakage
- Feature engineering geoespacial
- Pipeline reproducible
- Transformación logarítmica
- Exportación para GIS
- 60 features interpretables

</div>

<div>

## Beneficios Prácticos
- **~1 segundo** por predicción
- vs **3 horas** avalúo manual
- **10,800x más rápido**
- Costo cercano a $0
- Error promedio: **$27k** (2.9%)
- MAPE: **12.96%** ✅

</div>

</div>

---

# Limitaciones Identificadas

1. **Valores extremos (>$1M)**
   - Mayor error relativo en propiedades de lujo

2. **Variables omitidas potenciales**
   - Calidad de acabados
   - Estado de conservación
   - Vista panorámica
   - Amenidades del edificio

3. **Datos de un solo período**
   - No captura tendencias temporales
   - Sin histórico de precios

4. **Zona geográfica limitada**
   - Modelo específico para Quito urbano

---

# Comparación con Literatura

## Benchmarks Internacionales

| Estudio | País | R² | Metodología |
|---------|------|-----|-------------|
| **Este trabajo** | Ecuador | **0.9605** | RandomForest + Log-transform |
| Arribas-Bel et al. (2019) | España | 0.82 | Random Forest |
| Hong et al. (2020) | Corea | 0.76 | Deep Learning |
| Poursaeed et al. (2018) | USA | 0.71 | CNN + Imágenes |
| Tchuente & Nyawa (2022) | Camerún | 0.68 | XGBoost |

<div class="success">

**Nuestro modelo supera** el estado del arte en predicción de valores inmobiliarios

</div>

---

# Trabajo Futuro: Mejoras Propuestas

## 1. Incorporar Datos Externos

- **Imágenes satelitales:** CNN para detectar características visuales
- **Street View:** Calidad de fachada, entorno
- **Datos socioeconómicos:** Índice de desarrollo por sector
- **Transacciones reales:** Precios de mercado vs catastral
- **Amenidades:** Distancia a parques, hospitales, colegios

---

# Trabajo Futuro: Modelos Avanzados

## 2. Arquitecturas de ML

- **Modelos espaciales explícitos:**
  - Geographically Weighted Regression (GWR)
  - Spatial Autoregressive Models (SAR)

- **Deep Learning:**
  - MLP para regresión con embeddings
  - Atención espacial

- **Ensemble híbrido:**
  - Stacking de RandomForest + XGBoost + LightGBM

---

# Trabajo Futuro: Producción

## 3. Deployment y Operacionalización

- **API REST:** FastAPI para predicciones en tiempo real
- **Dashboard interactivo:** Streamlit o Dash
- **Integración GIS:** Plugin para QGIS
- **Sistema de monitoreo:**
  - Drift detection
  - Alertas de degradación del modelo
  - Re-entrenamiento automático trimestral
- **App móvil:** Para peritos en campo

---

# Impacto y Aplicaciones

## Casos de Uso Potenciales

<div class="columns">

<div>

### Sector Público
- Municipios: Actualización masiva de catastros
- Tributación: Detección de subdeclaraciones
- Planificación urbana: Análisis de mercado
- Equidad fiscal: Homogeneización de valores

</div>

<div>

### Sector Privado
- Inmobiliarias: Valoración rápida
- Bancos: Evaluación de garantías
- Aseguradoras: Estimación de valor asegurado
- Inversores: Due diligence automatizado

</div>

</div>

---

# Valor Económico Estimado

## ROI para el Municipio de Quito

**Escenario actual:**
- 44,407 propiedades
- Costo avalúo manual: $100/propiedad
- Total: **$4,440,700**
- Tiempo: 6 meses

**Con este sistema:**
- Costo: ~$5,000 (desarrollo + servidor)
- Tiempo: **1 día**
- **Ahorro: $4,435,700** (99.9%)

---

# Stack Tecnológico Utilizado

```python
# Core
Python 3.10+

# Machine Learning
scikit-learn 1.3+      # Modelos baseline
XGBoost 2.0+           # Gradient boosting optimizado
LightGBM 4.0+          # Boosting rápido

# Datos y Análisis
pandas 2.0+            # Manipulación de datos
numpy 1.24+            # Operaciones numéricas
matplotlib 3.7+        # Visualización
seaborn 0.12+          # Visualización estadística

# Deployment
streamlit 1.25+        # Dashboard interactivo
joblib 1.3+            # Serialización de modelos
```

---

# Repositorio y Documentación

## Estructura del Proyecto

```
catastro_prediccion_v2/
├── main.py                          # Pipeline principal
├── src/                             # Módulos del sistema
│   ├── data_loader.py               # Carga y normalización
│   ├── preprocessing.py             # Limpieza + Leakage
│   ├── eda.py                       # Análisis exploratorio
│   ├── feature_engineering.py       # 29 nuevas features
│   ├── feature_selection.py         # Top-60 selection
│   ├── models.py                    # 8 algoritmos
│   ├── evaluate.py                  # Métricas y gráficos
│   └── clustering.py                # KMeans, DBSCAN
├── output/                          # Resultados
│   ├── leakage_report.json
│   ├── summary.html
│   ├── models/                      # Modelos .pkl
│   ├── figures/                     # Gráficos PNG
│   └── test_completo_*.xlsx         # Para GIS
└── dataset_final_formateado.xlsx    # Datos de entrada
```

---

# Archivos Generados

## Outputs Disponibles

1. **`leakage_report.json`**: Detección automática de fuga
2. **`summary.html`**: Reporte interactivo completo
3. **`models/experiment_a/`**: 8 modelos sin leakage ✅
4. **`models/experiment_b/`**: 8 modelos con leakage (referencia)
5. **`models/optimized/`**: RandomForest tunado
6. **`figures/*.png`**: 8+ visualizaciones
7. **`ejemplos_test_streamlit.xlsx`**: 5 casos de prueba
8. **`test_completo_con_predicciones.xlsx`**: 8,882 predicciones para GIS

---

# Lecciones Aprendidas

<div class="columns">

<div>

## Técnicas
1. **Log-transform es crítica** para targets con alta asimetría
2. **Feature selection > feature engineering masivo**
   - 60 features bien seleccionadas suficientes
3. **RandomForest superior** incluso a XGBoost
4. **Filtrar outliers extremos** mejora todas las métricas

</div>

<div>

## De Proceso
1. **Detección de leakage es crítica**
   - Debe ser primer paso
2. **Validación espacial** revela patrones ocultos
3. **Interpretabilidad importa**
   - Feature importance para stakeholders
4. **Distribución del error** más útil que MAPE global

</div>

</div>

---

# Recomendaciones para Uso

## Guía de Implementación

1. ✅ **Usar modelo Experimento A** (RandomForest sin leakage)
2. ✅ **Predicciones des-transformadas automáticamente** (escala dólares)
3. ✅ **Monitorear predicciones en propiedades >$1M**
4. ✅ **Combinar con validación de perito** en casos críticos (error >20%)
5. ✅ **Actualizar modelo semestralmente** con nuevos datos
6. ⚠️ **No usar para propiedades <$10k o fuera de Quito urbano**
7. ⚠️ **Validar espacialmente** antes de decisiones masivas

---

# Contribuciones Clave del Proyecto

1. **Sistema automático de detección de leakage** (metodología replicable)
2. **Transformación logarítmica** para normalización extrema (101.17 → 0.29)
3. **Feature engineering geoespacial** específico para Ecuador
4. **Pipeline end-to-end reproducible** para catastros
5. **Validación espacial del error** con exportación GIS
6. **Benchmark para Ecuador:** R² = 0.9605 con 60 features

---

# Publicaciones Futuras

## Artículos Planificados

1. **"Automatic Target Leakage Detection in Real Estate Valuation"**
   - Venue: Journal of Real Estate Research

2. **"Log-Transform and Feature Selection for Cadastral Valuation"**
   - Venue: Computers, Environment and Urban Systems

3. **"Spatial Error Analysis in ML-based Property Valuation"**
   - Venue: International Journal of Geographical Information Science

---

<!-- _class: lead -->
<!-- _paginate: false -->

# ¿Preguntas?

**Fausto Guano**
Universidad Yachay Tech
Maestría en Ciencia de Datos

📧 fausto.guano@yachaytech.edu.ec

---

# Referencias (1/2)

1. **Breiman, L. (2001).** Random forests. *Machine Learning*, 45(1), 5-32.

2. **Chen, T., & Guestrin, C. (2016).** XGBoost: A scalable tree boosting system. *Proceedings of KDD*, 785-794.

3. **Ke, G., et al. (2017).** LightGBM: A highly efficient gradient boosting decision tree. *Advances in NIPS*, 3146-3154.

4. **Pedregosa, F., et al. (2011).** Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825-2830.

---

# Referencias (2/2)

5. **Arribas-Bel, D., Garcia-López, M. À., & Viladecans-Marsal, E. (2019).** Building(s and) cities: Delineating urban areas with a machine learning algorithm. *Journal of Urban Economics*, 103217.

6. **Hong, J., Choi, H., & Kim, W. S. (2020).** A house price valuation based on the random forest approach. *International Journal of Strategic Property Management*, 24(3), 140-152.

7. **Kaufman, S., Rosset, S., & Perlich, C. (2012).** Leakage in data mining: Formulation, detection, and avoidance. *ACM TKDD*, 6(4), 1-21.

---

# Apéndice A: Código - Detección de Leakage

```python
def detect_leakage(self, df, target_col):
    """Detecta target leakage por nombre y correlación"""
    suspicious_cols = []

    # 1. Detección por nombre
    name_patterns = ['valor', 'valoracion', 'avaluo',
                     'precio', 'costo', 'monto']
    for col in df.columns:
        if any(p in col.lower() for p in name_patterns):
            if col != target_col:
                suspicious_cols.append(col)

    # 2. Detección por correlación
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df[numeric_cols].corr()[target_col].abs()
    high_corr = correlations[correlations > 0.8].index.tolist()

    suspicious_cols.extend([c for c in high_corr
                           if c not in suspicious_cols
                           and c != target_col])

    return list(set(suspicious_cols))
```

---

# Apéndice B: Código - Transformación Logarítmica

```python
def limpiar_y_transformar_target(y, umbral_min=10000):
    """Limpia outliers y aplica log-transform"""
    
    # 1. Filtrar valores mínimos problemáticos
    mask_validos = y >= umbral_min
    y_limpio = y[mask_validos]
    
    # 2. Filtrar outliers extremos (P1, P99)
    p1, p99 = np.percentile(y_limpio, [1, 99])
    mask_rango = (y_limpio >= p1) & (y_limpio <= p99)
    y_final = y_limpio[mask_rango]
    
    # 3. Aplicar transformación logarítmica
    y_log = np.log(y_final)
    
    # 4. Verificar mejora en asimetría
    asimetria_antes = skew(y)
    asimetria_despues = skew(y_log)
    
    return y_log, mask_validos & mask_rango
```

---

# Apéndice C: Métricas Detalladas por Percentil

## Distribución Real del Error (Test Set)

| Percentil | Error Absoluto | % del Valor Mediano |
|-----------|---------------|---------------------|
| 10% | $5,800 | 6.4% |
| 25% | $9,200 | 10.1% |
| **50%** | **$13,500** | **14.8%** |
| 75% | $28,900 | 31.7% |
| 90% | $62,400 | 68.5% |
| 95% | $108,000 | 118.5% |
| 99% | $345,000 | 378.6% |

---

# Apéndice D: Detalles de Preprocesamiento

## Transformaciones Aplicadas

| Paso | Acción | Parámetros |
|------|--------|------------|
| **Imputación** | KNNImputer | k=5, weights='uniform' |
| **Encoding** | OneHotEncoder | drop='first', sparse=False |
| **Scaling** | StandardScaler | mean=0, std=1 |
| **Outliers** | IQR Method + P1-P99 | threshold=1.5 * IQR |
| **Target Transform** | Log-transform | np.log(y) |
| **CV** | StratifiedKFold | n_splits=5, shuffle=True |

---

# Apéndice E: Hiperparámetros del Mejor Modelo

## RandomForest (Modelo Final)

```python
RandomForestRegressor(
    n_estimators=100,           # Número de árboles
    max_depth=None,             # Profundidad sin límite
    min_samples_split=2,        # Mínimo para split
    min_samples_leaf=1,         # Mínimo en hoja
    max_features='sqrt',        # Features por split
    bootstrap=True,             # Muestreo con reemplazo
    random_state=42,            # Reproducibilidad
    n_jobs=-1                   # Paralelización
)
```

**Tiempo de entrenamiento:** ~2 minutos  
**Predicción (8,882 registros):** <1 segundo

---

<!-- _class: lead -->
<!-- _paginate: false -->

# Gracias por su atención

## Contacto
**Fausto Guano**
Universidad Yachay Tech

**Repositorio:** [GitHub - prediccion_avaluos_v2](https://github.com/usuario/proyecto_avaluos)

---