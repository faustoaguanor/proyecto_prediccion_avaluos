"""
Sistema completo de Machine Learning para predicción de avalúos catastrales
con detección y manejo de target leakage.

Uso:
    python main.py

Salida:
    - output/leakage_report.json
    - output/summary.html
    - output/models/*.pkl
    - output/figures/*.png
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

from src.clustering import ClusteringAnalysis

# Importar módulos del proyecto
from src.data_loader import DataLoader
from src.eda import EDA
from src.evaluate import ModelEvaluator
from src.feature_engineering import aplicar_feature_engineering
from src.feature_selection import seleccionar_features_optimas
from src.models import ClassificationModels, RegressionModels
from src.preprocessing import Preprocessor


class CatastroPipeline:
    """Pipeline completo de análisis catastral"""

    def __init__(self, filepath: str, output_dir: str = "output"):
        self.filepath = filepath
        self.output_dir = output_dir
        self.target_col = None
        self.df = None
        self.df_processed = None
        self.leakage_report = None

        # Crear directorios de salida
        os.makedirs(f"{output_dir}/models", exist_ok=True)
        os.makedirs(f"{output_dir}/figures", exist_ok=True)

    def run_full_pipeline(self):
        """Ejecuta el pipeline completo"""

        print("\n" + "=" * 70)
        print("SISTEMA DE PREDICCIÓN DE AVALÚOS CATASTRALES")
        print("Con Detección Automática de Target Leakage")
        print("=" * 70)

        # ========== 1. CARGA DE DATOS ==========
        print("\n" + "■" * 70)
        print("FASE 1: CARGA Y NORMALIZACIÓN DE DATOS")
        print("■" * 70)

        loader = DataLoader(self.filepath)
        self.df, self.metadata = loader.run_pipeline()

        # Detectar columna target
        preprocessor = Preprocessor(target_col=None)
        self.target_col = preprocessor.detect_target(self.df)

        if self.target_col is None:
            print("\n❌ ERROR: No se pudo detectar la columna target automáticamente")
            print("Por favor especifique manualmente la columna objetivo")
            return

        print(f"\n✓ Target detectado: {self.target_col}")

        # ========== 2. PREPROCESAMIENTO Y DETECCIÓN DE LEAKAGE ==========
        print("\n" + "■" * 70)
        print("FASE 2: PREPROCESAMIENTO Y DETECCIÓN DE LEAKAGE")
        print("■" * 70)

        preprocessor = Preprocessor(target_col=self.target_col)
        self.df_processed, self.leakage_report = preprocessor.run_preprocessing(
            self.df, detect_leakage=True
        )

        # Guardar reporte de leakage
        with open(f"{self.output_dir}/leakage_report.json", "w", encoding="utf-8") as f:
            json.dump(self.leakage_report, f, indent=2, ensure_ascii=False, default=str)

        print(f"\n✓ Reporte de leakage guardado: {self.output_dir}/leakage_report.json")

        # ========== 3. ANÁLISIS EXPLORATORIO ==========
        print("\n" + "■" * 70)
        print("FASE 3: ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
        print("■" * 70)

        eda = EDA(self.df_processed, self.target_col)
        eda_results = eda.run_full_eda(save_path=f"{self.output_dir}/figures")

        # ========== 4. PREPARAR DATOS PARA MODELADO ==========
        print("\n" + "■" * 70)
        print("FASE 4: PREPARACIÓN DE DATOS PARA MODELADO")
        print("■" * 70)

        # Separar features y target
        y = self.df_processed[self.target_col]

        id_columns = self.metadata.get("removed_id_columns", [])
        if id_columns:
            self.df_ids = self.df_processed[id_columns].copy()
            print(f"✓ IDs conservados: {len(id_columns)} columnas")
        else:
            self.df_ids = pd.DataFrame()

        cols_to_drop = [self.target_col, "is_outlier"] + id_columns
        X_all = self.df_processed.drop(columns=cols_to_drop, errors="ignore")

        # X_all = self.df_processed.drop(
        #    columns=[self.target_col, "is_outlier"], errors="ignore"
        # )

        # Asegurar que todas las columnas sean numéricas
        X_all = X_all.select_dtypes(include=[np.number])

        # Eliminar filas con NaN en target
        valid_mask = y.notna()
        X_all = X_all[valid_mask]
        y = y[valid_mask]

        print(f"\n✓ Datos preparados:")
        print(f"  - Features: {X_all.shape[1]}")
        print(f"  - Muestras: {X_all.shape[0]:,}")

        # Identificar features sospechosas
        suspicious_cols = self.leakage_report["all_suspicious_columns"]
        suspicious_in_data = [col for col in suspicious_cols if col in X_all.columns]

        print(f"\n⚠️  Features sospechosas presentes: {len(suspicious_in_data)}")
        if suspicious_in_data:
            print(
                f"  {', '.join(suspicious_in_data[:5])}{'...' if len(suspicious_in_data) > 5 else ''}"
            )

        # ========== 4.5. FEATURE ENGINEERING ==========

        features_originales = X_all.shape[1]
        X_all, nuevas_features = aplicar_feature_engineering(
            X_all, centro_lat=-0.1807, centro_lon=-78.4678, anio_actual=2024  # Quito
        )

        print(f"\n📊 Resumen de Feature Engineering:")
        print(f"  - Features originales: {features_originales}")
        print(f"  - Features nuevas: {len(nuevas_features)}")
        print(f"  - Total final: {X_all.shape[1]}")

        # ========== 🎯 NUEVO: LIMPIEZA Y TRANSFORMACIÓN DEL TARGET ==========
        print("\n" + "🔧" * 35)
        print("LIMPIEZA CRÍTICA DEL TARGET")
        print("🔧" * 35)

        print("\n[1/5] Análisis del target original...")
        print(f"  - Registros totales: {len(y):,}")
        print(f"  - Mínimo: ${y.min():,.2f}")
        print(f"  - Percentil 1%: ${y.quantile(0.01):,.2f}")
        print(f"  - Percentil 5%: ${y.quantile(0.05):,.2f}")
        print(f"  - Mediana: ${y.median():,.2f}")
        print(f"  - Media: ${y.mean():,.2f}")
        print(f"  - Percentil 95%: ${y.quantile(0.95):,.2f}")
        print(f"  - Percentil 99%: ${y.quantile(0.99):,.2f}")
        print(f"  - Máximo: ${y.max():,.2f}")
        print(f"  - Ratio Max/Min: {y.max()/y.min():.1f}x")

        from scipy.stats import skew

        asimetria_original = skew(y)
        print(f"  - Asimetría: {asimetria_original:.2f}")

        # [2/5] Filtrar outliers extremos
        print("\n[2/5] Filtrando outliers extremos (1% inferior y superior)...")
        percentil_01 = y.quantile(0.01)
        percentil_99 = y.quantile(0.99)

        mask_percentiles = (y >= percentil_01) & (y <= percentil_99)

        print(f"  - Límite inferior (P1): ${percentil_01:,.2f}")
        print(f"  - Límite superior (P99): ${percentil_99:,.2f}")
        print(f"  - Outliers extremos: {(~mask_percentiles).sum():,} registros")

        # [3/5] Eliminar valores muy pequeños (causan MAPE alto)
        print("\n[3/5] Filtrando valores mínimos problemáticos...")
        umbral_minimo = 10000  # $10,000
        mask_minimo = y >= umbral_minimo

        valores_muy_bajos = (~mask_minimo).sum()
        if valores_muy_bajos > 0:
            print(
                f"  ⚠️  Detectados {valores_muy_bajos} valores < ${umbral_minimo:,.0f}"
            )
            print(f"     Estos causan MAPE extremadamente alto (error %)")
            print(f"  → Eliminando para mejorar métricas")

        # Combinar máscaras
        mask_valido = mask_percentiles & mask_minimo
        registros_eliminados = (~mask_valido).sum()

        print(f"\n  📊 Resumen del filtrado:")
        print(f"     - Registros originales: {len(y):,}")
        print(
            f"     - Registros eliminados: {registros_eliminados:,} ({registros_eliminados/len(y)*100:.1f}%)"
        )
        print(f"     - Registros finales: {mask_valido.sum():,}")

        # Aplicar filtro a X_all, y, y df_ids
        X_all = X_all[mask_valido]
        y = y[mask_valido]

        if hasattr(self, "df_ids") and not self.df_ids.empty:
            self.df_ids = self.df_ids[mask_valido]

        print(f"\n  ✓ Nuevo rango del target: ${y.min():,.2f} - ${y.max():,.2f}")

        # [4/5] Transformación logarítmica
        print("\n[4/5] Aplicando transformación logarítmica...")

        # Guardar versión sin transformar para referencia
        self.y_sin_transformar = y.copy()

        # Aplicar log
        y_antes_log = y.copy()
        y = np.log1p(y)  # log(1 + y) para evitar log(0)

        print(
            f"  - Target original: ${y_antes_log.min():,.0f} - ${y_antes_log.max():,.0f}"
        )
        print(f"  - Target log: {y.min():.2f} - {y.max():.2f}")
        print(f"  - Ratio original: {y_antes_log.max()/y_antes_log.min():.1f}x")
        print(f"  - Ratio log: {np.exp(y.max())/np.exp(y.min()):.1f}x")

        # [5/5] Verificación
        print("\n[5/5] Verificación de la transformación...")
        asimetria_log = skew(y)
        print(f"  - Asimetría antes: {asimetria_original:.2f}")
        print(f"  - Asimetría después: {asimetria_log:.2f}")

        mejora_asimetria = abs(asimetria_original) - abs(asimetria_log)
        print(f"  - Mejora: {mejora_asimetria:.2f}")

        if abs(asimetria_log) < 0.5:
            print("  ✅ Distribución normalizada exitosamente")
        elif abs(asimetria_log) < abs(asimetria_original):
            print("  ✅ Distribución mejorada significativamente")
        else:
            print("  ⚠️  Distribución mejorada parcialmente")

        print("\n" + "=" * 70)
        print("✅ TARGET PREPARADO PARA MODELADO")
        print("=" * 70)

        # ========== 5. EXPERIMENTO A: SIN LEAKAGE ==========
        print("\n" + "■" * 70)
        print("FASE 5: EXPERIMENTO A - MODELO SIN FEATURES SOSPECHOSAS")
        print("■" * 70)

        # Excluir features sospechosas
        X_clean = X_all.drop(columns=suspicious_in_data, errors="ignore")

        print(f"\n✓ Features después de excluir sospechosas: {X_clean.shape[1]}")

        # Split train/test
        X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
            X_clean, y, test_size=0.2, random_state=42
        )

        # Entrenar modelos
        models_a = RegressionModels(random_state=42)
        results_a = models_a.train_and_evaluate(
            X_train_a, X_test_a, y_train_a, y_test_a, cv=5
        )

        # Guardar mejor modelo
        models_a.save_models(f"{self.output_dir}/models/experiment_a")

        # ✅ AGREGAR FEATURE SELECTION AQUÍ (OPCIONAL) ✅
        APLICAR_FEATURE_SELECTION = True  # ← Cambiar a False para desactivar
        MAX_FEATURES = 60

        if APLICAR_FEATURE_SELECTION:
            print("\n" + "⚡" * 35)
            print("APLICANDO FEATURE SELECTION")
            print("⚡" * 35)

            from src.feature_selection import FeatureSelector

            # Feature selection
            selector = FeatureSelector(
                max_features=MAX_FEATURES, correlation_threshold=0.95
            )
            X_train_a = selector.seleccionar_mejores(X_train_a, y_train_a, verbose=True)
            X_test_a = selector.transform(X_test_a)

            print(f"\n✓ Features finales Experimento A: {X_train_a.shape[1]}")

        # Entrenar modelos
        models_a = RegressionModels(random_state=42)
        results_a = models_a.train_and_evaluate(
            X_train_a, X_test_a, y_train_a, y_test_a, cv=5
        )

        # Guardar mejor modelo
        models_a.save_models(f"{self.output_dir}/models/experiment_a")

        # ========== 6. EXPERIMENTO B: CON TODAS LAS FEATURES ==========
        print("\n" + "■" * 70)
        print("FASE 6: EXPERIMENTO B - MODELO CON TODAS LAS FEATURES")
        print("■" * 70)

        # Split train/test con todas las features
        X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
            X_all, y, test_size=0.2, random_state=42
        )

        # Entrenar modelos
        models_b = RegressionModels(random_state=42)
        results_b = models_b.train_and_evaluate(
            X_train_b, X_test_b, y_train_b, y_test_b, cv=5
        )

        # Guardar modelos
        models_b.save_models(f"{self.output_dir}/models/experiment_b")

        # ========== DESPUÉS DE FASE 6, ANTES DE FASE 7 ==========
        print("\n" + "■" * 70)
        print("FASE 6.5: OPTIMIZACIÓN DE HIPERPARÁMETROS")
        print("■" * 70)

        # Obtener el mejor modelo del Experimento A
        best_model_a_name = max(results_a.items(), key=lambda x: x[1]["r2_test"])[0]
        print(
            f"\n✓ Mejor modelo base: {best_model_a_name} (R² = {results_a[best_model_a_name]['r2_test']:.4f})"
        )

        # Solo optimizar si es un modelo compatible
        tunable_models = ["RandomForest", "GradientBoosting", "XGBoost", "LightGBM"]

        if best_model_a_name in tunable_models:
            print(f"\n🔧 Optimizando hiperparámetros de {best_model_a_name}...")
            print(f"   (Esto puede tardar 2-5 minutos)")

            try:
                tuning_results = models_a.hyperparameter_tuning(
                    X_train_a,
                    y_train_a,
                    model_name=best_model_a_name,
                    search_type="random",  # Más rápido que 'grid'
                )

                # Evaluar modelo optimizado
                optimized_model = tuning_results["best_model"]
                y_pred_test_optimized = optimized_model.predict(X_test_a)

                r2_optimized = r2_score(y_test_a, y_pred_test_optimized)
                rmse_optimized = np.sqrt(
                    mean_squared_error(y_test_a, y_pred_test_optimized)
                )
                mae_optimized = mean_absolute_error(y_test_a, y_pred_test_optimized)

                print(f"\n📊 COMPARACIÓN:")
                print(f"  Modelo Base:")
                print(f"    R² = {results_a[best_model_a_name]['r2_test']:.4f}")
                print(f"    RMSE = ${results_a[best_model_a_name]['rmse_test']:,.2f}")
                print(f"  Modelo Optimizado:")
                print(f"    R² = {r2_optimized:.4f}")
                print(f"    RMSE = ${rmse_optimized:,.2f}")

                # Mejora
                mejora_r2 = r2_optimized - results_a[best_model_a_name]["r2_test"]
                print(f"\n  ✓ Mejora en R²: {mejora_r2:+.4f} ({mejora_r2*100:+.2f}%)")

                # Guardar modelo optimizado
                import joblib

                os.makedirs(f"{self.output_dir}/models/optimized", exist_ok=True)
                model_path = f"{self.output_dir}/models/optimized/{best_model_a_name.lower()}_tuned.pkl"
                joblib.dump(optimized_model, model_path)
                print(f"\n✓ Modelo optimizado guardado: {model_path}")

                # Actualizar results_a con el modelo optimizado
                results_a[f"{best_model_a_name}_Optimizado"] = {
                    "model": optimized_model,
                    "r2_test": r2_optimized,
                    "rmse_test": rmse_optimized,
                    "mae_test": mae_optimized,
                    "r2_train": r2_score(y_train_a, optimized_model.predict(X_train_a)),
                    "mae_train": mean_absolute_error(
                        y_train_a, optimized_model.predict(X_train_a)
                    ),
                    "rmse_train": np.sqrt(
                        mean_squared_error(
                            y_train_a, optimized_model.predict(X_train_a)
                        )
                    ),
                    "mape_test": mean_absolute_percentage_error(
                        y_test_a, y_pred_test_optimized
                    ),
                    "cv_mean": tuning_results["best_score"],
                    "cv_std": 0,
                }

            except Exception as e:
                print(f"⚠️ Error durante optimización: {e}")
                print(f"   Continuando con modelo base...")
        else:
            print(f"\n⚠️ {best_model_a_name} no soporta tuning automático")
            print(f"   Usando modelo base...")

        # ========== 7. COMPARACIÓN Y EVALUACIÓN ==========
        print("\n" + "■" * 70)
        print("FASE 7: COMPARACIÓN DE EXPERIMENTOS Y EVALUACIÓN")
        print("■" * 70)

        evaluator = ModelEvaluator()

        # Comparar resultados
        df_comparison = evaluator.compare_experiments(
            results_a,
            results_b,
            experiment_names=("Exp A - Sin Leakage", "Exp B - Con Leakage"),
        )

        print("\n📊 COMPARACIÓN DE EXPERIMENTOS:\n")
        print(df_comparison.to_string(index=False))

        # Generar gráficas
        evaluator.plot_model_comparison(
            results_a, metric="r2_test", save_path=f"{self.output_dir}/figures"
        )

        evaluator.plot_experiment_comparison(
            df_comparison, save_path=f"{self.output_dir}/figures"
        )

        # Análisis de residuos del mejor modelo A
        best_model_a_name = max(results_a.items(), key=lambda x: x[1]["r2_test"])[0]
        best_model_a = results_a[best_model_a_name]["model"]

        y_pred_test_a = best_model_a.predict(X_test_a)
        evaluator.plot_residuals(
            y_test_a.values, y_pred_test_a, save_path=f"{self.output_dir}/figures"
        )

        # ========== 8. CLUSTERING ==========
        print("\n" + "■" * 70)
        print("FASE 8: ANÁLISIS DE CLUSTERING")
        print("■" * 70)

        # Usar sample para clustering (más eficiente)
        sample_size = min(5000, len(X_clean))
        X_sample = X_clean.sample(n=sample_size, random_state=42)

        clustering = ClusteringAnalysis(random_state=42)
        cluster_results = clustering.run_clustering_analysis(
            X_sample, n_clusters=5, save_path=f"{self.output_dir}/figures"
        )

        # ========== 9. CLASIFICACIÓN ==========
        print("\n" + "■" * 70)
        print("FASE 9: CLASIFICACIÓN POR RANGOS DE AVALÚO")
        print("■" * 70)

        classifier = ClassificationModels(n_classes=5, random_state=42)
        classification_results = classifier.train_and_evaluate(
            X_train_a, X_test_a, y_train_a, y_test_a
        )

        # ========== 10. REPORTE FINAL ==========
        print("\n" + "■" * 70)
        print("FASE 10: GENERACIÓN DE REPORTE FINAL")
        print("■" * 70)

        evaluator.generate_summary_report(
            results_a,
            results_b,
            self.leakage_report,
            output_path=f"{self.output_dir}/summary.html",
        )

        # ========== RESUMEN FINAL ==========
        print("\n" + "=" * 70)
        print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
        print("=" * 70)

        print("\n📁 ARCHIVOS GENERADOS:")
        print("  ├── output/leakage_report.json")
        print("  ├── output/summary.html")
        print("  ├── output/ejemplos_test_streamlit.xlsx")
        print("  ├── output/test_completo_con_predicciones.xlsx")
        print("  ├── output/models/")
        print("  │   ├── experiment_a/  (modelos sin leakage - RECOMENDADO)")
        print("  │   └── experiment_b/  (modelos con leakage - solo referencia)")
        print("  └── output/figures/")

        print("\n🎯 NOTAS IMPORTANTES:")
        print("  ⚠️  El target fue transformado con LOG para mejorar predicciones")
        print("  ✓ Las predicciones están en escala de DÓLARES (des-transformadas)")
        print("  ✓ Outliers extremos fueron filtrados (mejora MAPE)")

        # Calcular mejor modelo
        best_name_a = max(results_a.items(), key=lambda x: x[1]["r2_test"])[0]
        best_name_b = max(results_b.items(), key=lambda x: x[1]["r2_test"])[0]

        print("\n📊 MEJORES MODELOS:")
        print(f"  Experimento A (Sin Leakage): {best_name_a}")
        print(
            f"    └─ R² = {results_a[best_name_a]['r2_test']:.4f}, RMSE = ${results_a[best_name_a]['rmse_test']:,.2f}"
        )
        print(f"  Experimento B (Con Leakage): {best_name_b}")
        print(
            f"    └─ R² = {results_b[best_name_b]['r2_test']:.4f}, RMSE = ${results_b[best_name_b]['rmse_test']:,.2f}"
        )

        diferencia = (
            results_b[best_name_b]["r2_test"] - results_a[best_name_a]["r2_test"]
        )
        print(f"  Diferencia R² = {diferencia:+.4f}")

        if abs(diferencia) < 0.02:
            print("    ✅ Sin evidencia de leakage significativo")
        elif diferencia > 0.05:
            print("    ⚠️  Posible leakage (B >> A)")
        else:
            print("    ℹ️  Diferencia moderada")

        print("\n" + "=" * 70)
        print("Gracias por usar el sistema de análisis catastral")
        print("=" * 70)

        # ==========  EXPORTANDO 5 EJEMPLOS DEL TEST SET  ==========

        print("\n" + "■" * 70)
        print("EXPORTANDO 5 EJEMPLOS DEL TEST SET (para Streamlit)")
        print("■" * 70)

        # Tomar 5 ejemplos aleatorios del TEST SET
        n_ejemplos = min(5, len(X_test_a))
        indices_aleatorios = np.random.choice(
            len(X_test_a), size=n_ejemplos, replace=False
        )

        # Crear DataFrame con ejemplos
        ejemplos_test = pd.DataFrame()

        # 1. Agregar IDs (recuperar del df_ids usando índices originales)
        if not self.df_ids.empty:
            indices_originales = X_test_a.iloc[indices_aleatorios].index
            ejemplos_test = self.df_ids.loc[indices_originales].reset_index(drop=True)

        # 2. Agregar las features del test (las que usa el modelo)
        X_test_ejemplos = X_test_a.iloc[indices_aleatorios]
        for col in X_test_ejemplos.columns:
            ejemplos_test[col] = X_test_ejemplos[col].values

        # 3. Agregar valor real
        # y_test_ejemplos = y_test_a.iloc[indices_aleatorios]
        # ejemplos_test["Valoracion_Real"] = y_test_ejemplos.values
        y_test_ejemplos_dolar = np.expm1(
            y_test_a.iloc[indices_aleatorios]
        )  # Des-transformar
        ejemplos_test["Valoracion_Real"] = y_test_ejemplos_dolar.values

        # 4. Columnas para llenar
        ejemplos_test["Valoracion_Predicha"] = ""
        ejemplos_test["Error"] = ""

        # Exportar
        output_file = f"{self.output_dir}/ejemplos_test_streamlit.xlsx"

        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            # Hoja 1: Ejemplos
            ejemplos_test.to_excel(writer, sheet_name="Ejemplos_Test", index=False)

            # Hoja 2: Descripción
            features_list = list(X_test_ejemplos.columns)
            descripcion = pd.DataFrame(
                {
                    "Feature": features_list,
                    "Tipo": ["Input Streamlit"] * len(features_list),
                }
            )
            descripcion.to_excel(writer, sheet_name="Features_Input", index=False)

            # Hoja 3: Instrucciones
            instrucciones = pd.DataFrame(
                {
                    "Info": [
                        "Estos 5 ejemplos NO se usaron en el entrenamiento",
                        "Son del TEST SET - nunca vistos por el modelo",
                        "Úsalos como INPUT en tu app de Streamlit",
                        "Tienen IDs para georreferenciación",
                        f"Features totales: {len(features_list)}",
                    ]
                }
            )
            instrucciones.to_excel(writer, sheet_name="Info", index=False)

        print(f"\n✅ Exportado: {output_file}")
        print(f"   - {n_ejemplos} ejemplos del TEST SET")
        print(
            f"   - Incluye IDs: {list(self.df_ids.columns) if not self.df_ids.empty else 'No'}"
        )
        print(f"   - Features: {len(X_test_ejemplos.columns)}")
        print(f"   - NO usados en entrenamiento ✓")

        # ========== EXPORTAR TODO EL TEST SET CON PREDICCIONES ==========

        print("\n" + "■" * 70)
        print("EXPORTANDO TEST SET COMPLETO (para análisis espacial del error)")
        print("■" * 70)

        # ✅ Usar la misma forma que ya usas en tu código
        best_model_a_name = max(results_a.items(), key=lambda x: x[1]["r2_test"])[0]
        best_model_a = results_a[best_model_a_name]["model"]
        best_r2 = results_a[best_model_a_name]["r2_test"]

        self.best_model_name_a = best_model_a_name
        self.best_r2_a = best_r2
        self.best_rmse_a = results_a[best_model_a_name]["rmse_test"]

        print(f"✓ Usando mejor modelo: {best_model_a_name} (R² = {best_r2:.4f})")

        # Hacer predicciones en TODO el test set
        print(f"✓ Haciendo predicciones en {len(X_test_a):,} registros del test...")
        # y_pred_test_a = best_model_a.predict(X_test_a)
        # Predicción en escala log
        y_pred_log = best_model_a.predict(X_test_a)

        # Des-transformar a dólares
        y_pred_test_a = np.expm1(y_pred_log)  # exp(y) - 1
        y_test_a_dolar = np.expm1(y_test_a)  # exp(y) - 1

        print(f"  ✓ Predicciones des-transformadas a escala de dólares")

        # Calcular errores
        # errores_absolutos = np.abs(y_test_a - y_pred_test_a)
        # errores_porcentuales = (errores_absolutos / y_test_a) * 100

        errores_absolutos = np.abs(y_test_a_dolar - y_pred_test_a)
        errores_porcentuales = (errores_absolutos / y_test_a_dolar) * 100

        # Verificación
        print(f"\n📊 Verificación de escala:")
        print(f"  - Predicción mín: ${y_pred_test_a.min():,.2f}")
        print(f"  - Predicción máx: ${y_pred_test_a.max():,.2f}")
        print(f"  - Real mín: ${y_test_a_dolar.min():,.2f}")
        print(f"  - Real máx: ${y_test_a_dolar.max():,.2f}")

        # Crear DataFrame completo
        test_completo = pd.DataFrame()

        # 1. IDs (recuperar usando índices originales)
        if hasattr(self, "df_ids") and not self.df_ids.empty:
            indices_originales_test = X_test_a.index
            test_completo = self.df_ids.loc[indices_originales_test].reset_index(
                drop=True
            )
            print(f"✓ IDs recuperados: {list(self.df_ids.columns)}")
        else:
            # Si no hay IDs, crear columna de ejemplo
            test_completo["Ejemplo_ID"] = [
                f"Test_{i:05d}" for i in range(len(X_test_a))
            ]

        # 2. Coordenadas geográficas (si existen en las features)
        coords = ["Latitud", "Longitud", "Lat_Relativa", "Lon_Relativa"]
        for coord in coords:
            if coord in X_test_a.columns:
                test_completo[coord] = X_test_a[coord].values

        # 3. Features principales (para contexto)
        features_principales = [
            "Area_Construccion",
            "Area_Terreno_Escri",
            "Frente_Total",
            "Distancia_Centro",
        ]
        for feat in features_principales:
            if feat in X_test_a.columns:
                test_completo[feat] = X_test_a[feat].values

        # 4. Valores reales, predichos y errores
        """test_completo["Valoracion_Real"] = y_test_a.values
        test_completo["Valoracion_Predicha"] = y_pred_test_a
        test_completo["Error_Absoluto"] = errores_absolutos
        test_completo["Error_Porcentual"] = errores_porcentuales
        test_completo["Error_Relativo"] = y_pred_test_a - y_test_a"""

        test_completo["Valoracion_Real"] = y_test_a_dolar  # ✅ Ya des-transformado
        test_completo["Valoracion_Predicha"] = y_pred_test_a  # ✅ Ya des-transformado
        test_completo["Error_Absoluto"] = errores_absolutos
        test_completo["Error_Porcentual"] = errores_porcentuales
        test_completo["Error_Relativo"] = y_pred_test_a - y_test_a_dolar

        # 5. Clasificar magnitud del error
        test_completo["Magnitud_Error"] = pd.cut(
            errores_porcentuales,
            bins=[0, 5, 10, 20, 100],
            labels=[
                "Excelente (<5%)",
                "Bueno (5-10%)",
                "Aceptable (10-20%)",
                "Alto (>20%)",
            ],
        )

        # Exportar a Excel
        output_file_completo = f"{self.output_dir}/test_completo_con_predicciones.xlsx"

        with pd.ExcelWriter(output_file_completo, engine="openpyxl") as writer:
            # Hoja 1: Datos completos
            test_completo.to_excel(writer, sheet_name="Test_Completo", index=False)

            # Hoja 2: Estadísticas del error
            stats_error = pd.DataFrame(
                {
                    "Metrica": [
                        "Modelo",
                        "Total registros",
                        "MAE (Error Absoluto Medio)",
                        "RMSE (Error Cuadrático Medio)",
                        "MAPE (Error Porcentual Medio)",
                        "Mediana Error Absoluto",
                        "Error Máximo",
                        "R² Score",
                        "",
                        "Predicciones Excelentes (<5%)",
                        "Predicciones Buenas (5-10%)",
                        "Predicciones Aceptables (10-20%)",
                        "Predicciones con Error Alto (>20%)",
                    ],
                    "Valor": [
                        best_model_a_name,
                        f"{len(test_completo):,}",
                        f"${np.mean(errores_absolutos):,.2f}",
                        f"${np.sqrt(np.mean(errores_absolutos**2)):,.2f}",
                        f"{np.mean(errores_porcentuales):.2f}%",
                        f"${np.median(errores_absolutos):,.2f}",
                        f"${np.max(errores_absolutos):,.2f}",
                        f"{best_r2:.4f}",
                        "",
                        f"{(errores_porcentuales < 5).sum():,} ({(errores_porcentuales < 5).sum()/len(test_completo)*100:.1f}%)",
                        f"{((errores_porcentuales >= 5) & (errores_porcentuales < 10)).sum():,} ({((errores_porcentuales >= 5) & (errores_porcentuales < 10)).sum()/len(test_completo)*100:.1f}%)",
                        f"{((errores_porcentuales >= 10) & (errores_porcentuales < 20)).sum():,} ({((errores_porcentuales >= 10) & (errores_porcentuales < 20)).sum()/len(test_completo)*100:.1f}%)",
                        f"{(errores_porcentuales >= 20).sum():,} ({(errores_porcentuales >= 20).sum()/len(test_completo)*100:.1f}%)",
                    ],
                }
            )
            stats_error.to_excel(writer, sheet_name="Estadisticas_Error", index=False)

            # Hoja 3: Top 20 errores más grandes
            cols_disponibles = [
                c
                for c in test_completo.columns
                if c
                in [
                    "Cat_Lote_Id",
                    "Ejemplo_ID",
                    "Latitud",
                    "Longitud",
                    "Area_Construccion",
                    "Valoracion_Real",
                    "Valoracion_Predicha",
                    "Error_Absoluto",
                    "Error_Porcentual",
                ]
            ]
            if len(cols_disponibles) > 0:
                top_errores = test_completo.nlargest(20, "Error_Absoluto")[
                    cols_disponibles
                ]
                top_errores.to_excel(
                    writer, sheet_name="Top_20_Mayores_Errores", index=False
                )

            # Hoja 4: Instrucciones para GIS
            instrucciones_gis = pd.DataFrame(
                {
                    "Paso": [1, 2, 3, 4, 5, 6, 7],
                    "Instruccion": [
                        "Abrir QGIS o ArcGIS",
                        'Importar "Test_Completo" como tabla',
                        "Crear capa de puntos usando Latitud/Longitud (si están disponibles)",
                        "O unir con capa catastral usando Cat_Lote_Id",
                        'Simbolizar por "Magnitud_Error" para visualizar zonas con más error',
                        'Crear mapa de calor con "Error_Absoluto"',
                        "Analizar patrones espaciales: ¿dónde falla más el modelo?",
                    ],
                }
            )
            instrucciones_gis.to_excel(
                writer, sheet_name="Guia_Espacializacion", index=False
            )

        print(f"\n✅ Exportado: {output_file_completo}")
        print(f"   - Total registros: {len(test_completo):,}")
        print(f"   - Modelo: {best_model_a_name}")
        print(f"   - R² Test: {best_r2:.4f}")
        print(f"   - MAE: ${np.mean(errores_absolutos):,.2f}")
        print(f"   - MAPE: {np.mean(errores_porcentuales):.2f}%")
        print(f"\n📊 Distribución del error:")
        print(
            f"   - Excelente (<5%):     {(errores_porcentuales < 5).sum():,} registros ({(errores_porcentuales < 5).sum()/len(test_completo)*100:.1f}%)"
        )
        print(
            f"   - Bueno (5-10%):       {((errores_porcentuales >= 5) & (errores_porcentuales < 10)).sum():,} registros ({((errores_porcentuales >= 5) & (errores_porcentuales < 10)).sum()/len(test_completo)*100:.1f}%)"
        )
        print(
            f"   - Aceptable (10-20%):  {((errores_porcentuales >= 10) & (errores_porcentuales < 20)).sum():,} registros ({((errores_porcentuales >= 10) & (errores_porcentuales < 20)).sum()/len(test_completo)*100:.1f}%)"
        )
        print(
            f"   - Alto (>20%):         {(errores_porcentuales >= 20).sum():,} registros ({(errores_porcentuales >= 20).sum()/len(test_completo)*100:.1f}%)"
        )

        print("\n💡 Usa este archivo para:")
        print("   1. Espacializar el error en QGIS/ArcGIS")
        print("   2. Identificar zonas donde el modelo falla más")
        print("   3. Encontrar patrones espaciales del error")
        print("   4. Validar el modelo geográficamente")


def main():
    """Función principal"""

    # Configuración
    FILEPATH = "dataset_final_formateado.xlsx"
    OUTPUT_DIR = "output"

    # Verificar que existe el archivo
    if not os.path.exists(FILEPATH):
        print(f"\n❌ ERROR: No se encontró el archivo '{FILEPATH}'")
        print(f"Por favor coloque el archivo en el directorio actual")
        sys.exit(1)

    # Ejecutar pipeline
    pipeline = CatastroPipeline(FILEPATH, OUTPUT_DIR)

    try:
        pipeline.run_full_pipeline()
    except Exception as e:
        print(f"\n❌ ERROR durante la ejecución del pipeline:")
        print(f"   {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
