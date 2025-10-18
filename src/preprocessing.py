# src/preprocessing.py - ACTUALIZADO PARA VALORACION_TERRENO

import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


class LeakageDetector:
    """Detecta features que pueden filtrar información del target"""

    LEAKAGE_PATTERNS = [
        "valor",
        "aiva",  # ✅ Incluye AIVA
    ]

    def __init__(self, target_col: str):
        self.target_col = target_col
        self.leakage_by_name = []
        self.leakage_by_stats = []
        self.report = {}

    def detect_by_name(self, columns: List[str]) -> List[str]:
        """Detecta columnas sospechosas por nombre"""
        suspicious = []

        for col in columns:
            col_lower = col.lower()

            # Excluir el target mismo
            if col == self.target_col:
                continue

            # Buscar patrones
            for pattern in self.LEAKAGE_PATTERNS:
                if pattern in col_lower:
                    suspicious.append(col)
                    break

        self.leakage_by_name = suspicious
        print(f"\n⚠ Detectadas {len(suspicious)} columnas sospechosas por NOMBRE:")
        for col in suspicious:
            print(f"  - {col}")

        return suspicious

    def detect_by_correlation(
        self, df: pd.DataFrame, threshold: float = 0.8
    ) -> List[str]:
        """Detecta columnas con correlación muy alta con el target"""
        suspicious = []

        if self.target_col not in df.columns:
            return suspicious

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != self.target_col]

        correlations = {}

        for col in numeric_cols:
            try:
                # Filtrar valores válidos
                valid_mask = df[col].notna() & df[self.target_col].notna()
                if valid_mask.sum() < 10:  # Necesitamos al menos 10 puntos
                    continue

                col_data = df.loc[valid_mask, col]
                target_data = df.loc[valid_mask, self.target_col]

                # Pearson
                pearson_corr = col_data.corr(target_data)

                # Spearman (más robusto a outliers)
                spearman_corr, p_value = stats.spearmanr(col_data, target_data)

                correlations[col] = {
                    "pearson": pearson_corr,
                    "spearman": spearman_corr,
                    "p_value": p_value,
                }

                # Marcar si correlación muy alta
                if abs(spearman_corr) > threshold and p_value < 0.01:
                    suspicious.append(col)
            except Exception as e:
                continue

        self.leakage_by_stats = suspicious
        self.report["correlations"] = correlations

        print(
            f"\n⚠ Detectadas {len(suspicious)} columnas con CORRELACIÓN > {threshold}:"
        )
        for col in suspicious:
            if col in correlations:
                corr = correlations[col]["spearman"]
                print(f"  - {col}: {corr:.3f}")

        return suspicious

    def get_all_suspicious(self) -> List[str]:
        """Retorna lista única de todas las columnas sospechosas"""
        all_suspicious = list(set(self.leakage_by_name + self.leakage_by_stats))
        return all_suspicious

    def generate_report(self) -> Dict:
        """Genera reporte completo de leakage"""
        report = {
            "target_column": self.target_col,
            "detection_methods": {
                "by_name": {
                    "count": len(self.leakage_by_name),
                    "columns": self.leakage_by_name,
                },
                "by_correlation": {
                    "count": len(self.leakage_by_stats),
                    "columns": self.leakage_by_stats,
                },
            },
            "all_suspicious_columns": self.get_all_suspicious(),
            "total_suspicious": len(self.get_all_suspicious()),
            "recommendation": "Excluir estas columnas del modelo de producción (Experimento A). Comparar con Experimento B que las incluye.",
        }

        if "correlations" in self.report:
            report["correlations"] = self.report["correlations"]

        if "importances" in self.report:
            report["importances"] = self.report["importances"]

        return report


class Preprocessor:
    """Preprocesamiento completo con detección de leakage - ACTUALIZADO"""

    def __init__(self, target_col: str):
        self.target_col = target_col
        self.leakage_detector = LeakageDetector(target_col)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputers = {}

    @staticmethod
    def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Limpia y reconstruye DataFrame para evitar problemas de indexación"""
        df = df.reset_index(drop=True)
        df_clean = pd.DataFrame()
        for col in df.columns:
            df_clean[col] = df[col].values.copy()
        return df_clean

    def detect_target(self, df: pd.DataFrame) -> str:
        """
        Detecta automáticamente la columna target.
        ✅ ACTUALIZADO para detectar Valoracion_Terreno
        """
        target_patterns = [
            "valoracion_terreno",  # ✅ NUEVO: Primera prioridad
            "valoracion_total",
            "valor_total",
            "avaluo_total",
            "valoracion",
            "avaluo",
        ]

        for pattern in target_patterns:
            matching = [col for col in df.columns if pattern.lower() in col.lower()]
            if matching:
                print(f"✓ Target detectado: {matching[0]}")
                return matching[0]

        return None

    def handle_missing_values(
        self, df: pd.DataFrame, strategy: str = "auto"
    ) -> pd.DataFrame:
        """Imputa valores faltantes según el tipo de columna"""
        df_imputed = self.clean_dataframe(df)

        numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns
        categorical_cols = df_imputed.select_dtypes(include=["object"]).columns

        for col in numeric_cols:
            missing_pct = df_imputed[col].isnull().mean()
            if missing_pct > 0:
                if df_imputed[col].dropna().empty:
                    print(f"⚠️ Columna numérica '{col}' completamente vacía.")
                    continue
                imputer = SimpleImputer(strategy="median")
                imputed_values = imputer.fit_transform(df_imputed[[col]])
                if imputed_values.size == 0:
                    print(f"⚠️ Imputación vacía para '{col}'.")
                    continue
                df_imputed[col] = imputed_values.ravel()
                self.imputers[col] = imputer

        for col in categorical_cols:
            missing_pct = df_imputed[col].isnull().mean()
            if missing_pct > 0:
                if df_imputed[col].dropna().empty:
                    print(f"⚠️ Columna categórica '{col}' completamente vacía.")
                    continue
                imputer = SimpleImputer(strategy="most_frequent")
                imputed_values = imputer.fit_transform(df_imputed[[col]])
                if imputed_values.size == 0:
                    print(f"⚠️ Imputación vacía para '{col}'.")
                    continue
                df_imputed[col] = imputed_values.ravel()
                self.imputers[col] = imputer

        print(f"✓ Imputación completada: {len(self.imputers)} columnas procesadas")
        return self.clean_dataframe(df_imputed)

    def encode_categorical(
        self, df: pd.DataFrame, max_cardinality: int = 10
    ) -> pd.DataFrame:
        """Codifica variables categóricas"""
        df_encoded = self.clean_dataframe(df)
        categorical_cols = df_encoded.select_dtypes(include=["object"]).columns
        encoded_count = 0

        for col in categorical_cols:
            n_unique = df_encoded[col].nunique()

            if n_unique <= max_cardinality:
                try:
                    dummies = pd.get_dummies(
                        df_encoded[col], prefix=col, drop_first=True, dtype=int
                    )
                    df_encoded = pd.concat([df_encoded, dummies], axis=1)
                    df_encoded.drop(col, axis=1, inplace=True)
                    encoded_count += 1
                except Exception as e:
                    print(f"    ⚠ Error codificando {col}: {str(e)[:50]}")
            else:
                try:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    self.label_encoders[col] = le
                    encoded_count += 1
                except Exception as e:
                    print(f"    ⚠ Error codificando {col}: {str(e)[:50]}")

        print(
            f"✓ Codificación completada: {encoded_count} columnas categóricas procesadas"
        )
        return self.clean_dataframe(df_encoded)

    def detect_outliers_catastral(
        self, df: pd.DataFrame, method: str = "iqr"
    ) -> pd.DataFrame:
        """Detecta outliers específicamente adaptado para datos catastrales"""
        print(f"  Iniciando detección de outliers (método: {method})...")
        df_clean = self.clean_dataframe(df)
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()

        # Excluir columnas específicas
        exclude_cols = ["alicuota", "numero_pisos", "anio_construccion", "anio_emision"]
        numeric_cols = [
            c
            for c in numeric_cols
            if not any(excl in c.lower() for excl in exclude_cols)
        ]

        outlier_mask = pd.Series([False] * len(df_clean), dtype=bool)
        outlier_counts = {}

        if method == "iqr":
            print(f"  Analizando {len(numeric_cols)} columnas numéricas con IQR...")
            for col in numeric_cols:
                try:
                    col_data = df_clean[col].dropna()
                    if len(col_data) < 10:
                        continue

                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1

                    if IQR > 0:
                        lower_bound = Q1 - 3 * IQR
                        upper_bound = Q3 + 3 * IQR
                        col_outliers = (df_clean[col] < lower_bound) | (
                            df_clean[col] > upper_bound
                        )
                        col_outliers = col_outliers.fillna(False)
                        n_outliers_col = col_outliers.sum()
                        if n_outliers_col > 0:
                            outlier_counts[col] = n_outliers_col
                        outlier_mask = outlier_mask | col_outliers
                except Exception as e:
                    print(f"    ⚠ Error en columna {col}: {str(e)[:50]}")
                    continue

        n_outliers = outlier_mask.sum()
        pct_outliers = (n_outliers / len(df_clean)) * 100
        print(
            f"✓ Outliers detectados ({method}): {n_outliers} filas ({pct_outliers:.1f}%)"
        )

        if outlier_counts and method == "iqr":
            print(f"  Top 5 columnas con más outliers:")
            sorted_counts = sorted(
                outlier_counts.items(), key=lambda x: x[1], reverse=True
            )[:5]
            for col, count in sorted_counts:
                print(f"    - {col}: {count} outliers")

        df_clean["is_outlier"] = outlier_mask.astype(int)  # ✅ Convertir a 0/1
        return df_clean

    def final_cleanup_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpieza final: asegura que NO haya NaN en columnas numéricas"""
        print("\n  [Limpieza Final] Verificando NaN residuales...")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        if categorical_cols:
            print(
                f"  ⚠ Eliminando {len(categorical_cols)} columnas categóricas residuales"
            )
            df = df.drop(columns=categorical_cols)

        nan_counts = df[numeric_cols].isnull().sum() if numeric_cols else pd.Series()
        total_nan = nan_counts.sum()

        if total_nan > 0:
            print(f"  ⚠ Detectados {total_nan} NaN en columnas numéricas")
            cols_with_nan = nan_counts[nan_counts > 0].sort_values(ascending=False)
            if len(cols_with_nan) > 0:
                print(f"  Columnas con NaN:")
                for col, count in cols_with_nan.head(5).items():
                    pct = (count / len(df)) * 100
                    print(f"    - {col}: {count} NaN ({pct:.1f}%)")

            cols_to_drop = []
            for col in numeric_cols:
                if col in df.columns and df[col].isnull().any():
                    nan_pct = df[col].isnull().sum() / len(df)
                    if nan_pct > 0.9:
                        cols_to_drop.append(col)

            if cols_to_drop:
                print(
                    f"  ✗ Eliminando {len(cols_to_drop)} columnas casi vacías (>90% NaN)"
                )
                df = df.drop(columns=cols_to_drop)

            remaining_nan = df.select_dtypes(include=[np.number]).isnull().sum().sum()
            if remaining_nan > 0:
                print(
                    f"  ✓ Llenando {remaining_nan} NaN con 0 (válido: 0 = 'no tiene')"
                )
                numeric_cols_remaining = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols_remaining] = df[numeric_cols_remaining].fillna(0)

            print(f"  ✓ Limpieza final completada")
        else:
            print(f"  ✓ No hay NaN residuales")

        final_nan = df.isnull().sum().sum()
        if final_nan > 0:
            print(f"  ⚠ ADVERTENCIA: Aún quedan {final_nan} NaN")

        return df

    def run_preprocessing(
        self, df: pd.DataFrame, detect_leakage: bool = True
    ) -> Tuple[pd.DataFrame, Dict]:
        """Pipeline completo de preprocesamiento"""
        print("\n" + "=" * 70)
        print("INICIANDO PREPROCESAMIENTO")
        print("=" * 70)

        df_processed = self.clean_dataframe(df)

        if detect_leakage:
            print("\n[1/5] Detección de Target Leakage...")
            self.leakage_detector.detect_by_name(df_processed.columns)
            if self.target_col in df_processed.columns:
                self.leakage_detector.detect_by_correlation(df_processed)

        print("\n[2/5] Imputación de valores faltantes...")
        df_processed = self.handle_missing_values(df_processed)

        print("\n[3/5] Codificación de variables categóricas...")
        df_processed = self.encode_categorical(df_processed)

        print("\n[4/5] Detección de outliers...")
        try:
            df_processed = self.detect_outliers_catastral(df_processed, method="iqr")
        except Exception as e:
            print(f"⚠ Error inesperado en detección de outliers: {e}")
            df_processed["is_outlier"] = 0

        if "is_outlier" in df_processed.columns:
            df_processed["is_outlier"] = df_processed["is_outlier"].astype(int)

        print("\n[5/5] Generando reporte de leakage...")
        leakage_report = self.leakage_detector.generate_report()

        df_processed = self.clean_dataframe(df_processed)
        df_processed = self.final_cleanup_numeric(df_processed)

        print(f"\n✓ Preprocesamiento completado")
        print(f"  - Shape final: {df_processed.shape}")
        print(f"  - NaN totales: {df_processed.isnull().sum().sum()}")
        print(
            f"  - Columnas sospechosas detectadas: {len(leakage_report['all_suspicious_columns'])}"
        )

        return df_processed, leakage_report
