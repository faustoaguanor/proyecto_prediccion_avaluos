"""
Feature Engineering para Avalúos Catastrales
Características específicas para bienes raíces e inmobiliario
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class RealEstateFeatureEngineer:
    """
    Feature Engineering especializado para datos inmobiliarios/catastrales
    """

    def __init__(
        self,
        centro_lat: float = -0.1807,
        centro_lon: float = -78.4678,
        anio_actual: int = 2024,
    ):
        """
        Args:
            centro_lat: Latitud del centro de referencia (default: centro de Quito)
            centro_lon: Longitud del centro de referencia (default: centro de Quito)
            anio_actual: Año actual para cálculos de edad
        """
        self.centro_lat = centro_lat
        self.centro_lon = centro_lon
        self.anio_actual = anio_actual
        self.features_creadas = []

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica todas las transformaciones de feature engineering

        Args:
            X: DataFrame con features originales

        Returns:
            DataFrame con features originales + nuevas features
        """
        X = X.copy()

        print("\n" + "■" * 70)
        print("FEATURE ENGINEERING - BIENES RAÍCES")
        print("■" * 70)

        # 1. Features de Áreas y Construcción
        X = self._crear_features_areas(X)

        # 2. Features Geoespaciales
        X = self._crear_features_geoespaciales(X)

        # 3. Features de Influencias
        X = self._crear_features_influencias(X)

        # 4. Features de Edad y Tiempo
        X = self._crear_features_temporales(X)

        # 5. Features de Regulación Urbana
        X = self._crear_features_regulacion(X)

        # Resumen
        print(f"\n{'='*70}")
        print(f"✓ Feature Engineering completado")
        print(f"  - Features creadas: {len(self.features_creadas)}")
        print(f"  - Total features: {X.shape[1]}")
        print(f"{'='*70}\n")

        return X

    def _crear_features_areas(self, X: pd.DataFrame) -> pd.DataFrame:
        """Crea features relacionadas con áreas y construcción"""
        print("\n[1/5] Creando features de áreas y construcción...")
        count = 0

        # Ratio Construcción/Terreno
        if "Area_Construccion" in X.columns and "Area_Terreno_Escri" in X.columns:
            X["Ratio_Construccion_Terreno"] = X["Area_Construccion"] / (
                X["Area_Terreno_Escri"] + 1
            )
            X["Area_Total"] = X["Area_Construccion"] + X["Area_Terreno_Escri"]
            X["Area_No_Construida"] = X["Area_Terreno_Escri"] - X["Area_Construccion"]
            X["Area_No_Construida"] = X["Area_No_Construida"].clip(lower=0)
            self.features_creadas.extend(
                ["Ratio_Construccion_Terreno", "Area_Total", "Area_No_Construida"]
            )
            count += 3
            print(f"  ✓ Ratio_Construccion_Terreno, Area_Total, Area_No_Construida")

        # Profundidad del terreno
        if "Frente_Total" in X.columns and "Area_Terreno_Escri" in X.columns:
            X["Profundidad_Estimada"] = X["Area_Terreno_Escri"] / (
                X["Frente_Total"] + 1
            )
            X["Ratio_Frente_Area"] = X["Frente_Total"] / (X["Area_Terreno_Escri"] + 1)
            self.features_creadas.extend(["Profundidad_Estimada", "Ratio_Frente_Area"])
            count += 2
            print(f"  ✓ Profundidad_Estimada, Ratio_Frente_Area")

        # Área por piso
        if "Pisos_PUGS" in X.columns and "Area_Construccion" in X.columns:
            X["Area_Por_Piso"] = X["Area_Construccion"] / (X["Pisos_PUGS"] + 1)
            self.features_creadas.append("Area_Por_Piso")
            count += 1
            print(f"  ✓ Area_Por_Piso")

        print(f"  → {count} features creadas")
        return X

    def _crear_features_geoespaciales(self, X: pd.DataFrame) -> pd.DataFrame:
        """Crea features geoespaciales y de ubicación"""
        print("\n[2/5] Creando features geoespaciales...")
        count = 0

        if "Latitud" in X.columns and "Longitud" in X.columns:
            # Distancia euclidiana al centro
            X["Distancia_Centro"] = np.sqrt(
                (X["Latitud"] - self.centro_lat) ** 2
                + (X["Longitud"] - self.centro_lon) ** 2
            )

            # Distancia Manhattan (más realista para ciudades)
            X["Distancia_Centro_Manhattan"] = np.abs(
                X["Latitud"] - self.centro_lat
            ) + np.abs(X["Longitud"] - self.centro_lon)

            # Cuadrantes
            X["Es_Norte"] = (X["Latitud"] > self.centro_lat).astype(int)
            X["Es_Este"] = (X["Longitud"] > self.centro_lon).astype(int)
            X["Cuadrante"] = X["Es_Norte"] * 2 + X["Es_Este"]  # 0=SO, 1=SE, 2=NO, 3=NE

            # Coordenadas relativas
            X["Lat_Relativa"] = X["Latitud"] - self.centro_lat
            X["Lon_Relativa"] = X["Longitud"] - self.centro_lon

            self.features_creadas.extend(
                [
                    "Distancia_Centro",
                    "Distancia_Centro_Manhattan",
                    "Es_Norte",
                    "Es_Este",
                    "Cuadrante",
                    "Lat_Relativa",
                    "Lon_Relativa",
                ]
            )
            count = 7
            print(f"  ✓ Distancia_Centro, Cuadrantes, Coordenadas relativas")

        print(f"  → {count} features creadas")
        return X

    def _crear_features_influencias(self, X: pd.DataFrame) -> pd.DataFrame:
        """Crea features agregadas de influencias"""
        print("\n[3/5] Creando features de influencias combinadas...")
        count = 0

        # Buscar columnas de influencias normalizadas
        influencias = [col for col in X.columns if "Infl_" in col and "_Norm" in col]

        if len(influencias) > 0:
            # Estadísticas de influencias
            X["Influencia_Total"] = X[influencias].sum(axis=1)
            X["Influencia_Media"] = X[influencias].mean(axis=1)
            X["Influencia_Max"] = X[influencias].max(axis=1)
            X["Influencia_Min"] = X[influencias].min(axis=1)
            X["Influencia_Std"] = X[influencias].std(axis=1)

            # Número de influencias significativas (>0.5)
            X["N_Influencias_Altas"] = (X[influencias] > 0.5).sum(axis=1)

            self.features_creadas.extend(
                [
                    "Influencia_Total",
                    "Influencia_Media",
                    "Influencia_Max",
                    "Influencia_Min",
                    "Influencia_Std",
                    "N_Influencias_Altas",
                ]
            )
            count = 6
            print(f"  ✓ Estadísticas de {len(influencias)} influencias")
        else:
            print(f"  ⚠ No se encontraron columnas de influencia")

        print(f"  → {count} features creadas")
        return X

    def _crear_features_temporales(self, X: pd.DataFrame) -> pd.DataFrame:
        """Crea features relacionadas con tiempo y edad"""
        print("\n[4/5] Creando features temporales...")
        count = 0

        if "Anio_Construccion" in X.columns:
            # Edad de la construcción
            X["Edad_Construccion"] = self.anio_actual - X["Anio_Construccion"]
            X["Edad_Construccion"] = X["Edad_Construccion"].clip(lower=0)

            # Categorías de edad
            X["Es_Nuevo"] = (X["Edad_Construccion"] <= 5).astype(int)
            X["Es_Moderno"] = (
                (X["Edad_Construccion"] > 5) & (X["Edad_Construccion"] <= 20)
            ).astype(int)
            X["Es_Viejo"] = (X["Edad_Construccion"] > 20).astype(int)

            # Década de construcción
            X["Decada_Construccion"] = (X["Anio_Construccion"] // 10) * 10

            self.features_creadas.extend(
                [
                    "Edad_Construccion",
                    "Es_Nuevo",
                    "Es_Moderno",
                    "Es_Viejo",
                    "Decada_Construccion",
                ]
            )
            count = 5
            print(f"  ✓ Edad_Construccion, categorías de edad, década")

        print(f"  → {count} features creadas")
        return X

    def _crear_features_regulacion(self, X: pd.DataFrame) -> pd.DataFrame:
        """Crea features de regulación urbana"""
        print("\n[5/5] Creando features de regulación urbana...")
        count = 0

        # COS (Coeficiente de Ocupación del Suelo)
        if "Cos_PUGS" in X.columns:
            X["Cos_PUGS_Pct"] = X["Cos_PUGS"] / 100
            X["Cos_Utilizado"] = (
                X["Ratio_Construccion_Terreno"]
                if "Ratio_Construccion_Terreno" in X.columns
                else 0
            )
            X["Margen_COS"] = X["Cos_PUGS_Pct"] - X["Cos_Utilizado"]
            self.features_creadas.extend(
                ["Cos_PUGS_Pct", "Cos_Utilizado", "Margen_COS"]
            )
            count += 3
            print(f"  ✓ Cos_PUGS_Pct, Cos_Utilizado, Margen_COS")

        # CUS (Coeficiente de Utilización del Suelo)
        if "Cus_PUGS" in X.columns:
            X["Cus_PUGS_Pct"] = X["Cus_PUGS"] / 100
            self.features_creadas.append("Cus_PUGS_Pct")
            count += 1
            print(f"  ✓ Cus_PUGS_Pct")

        # Potencial constructivo
        if (
            "Pisos_PUGS" in X.columns
            and "Area_Construccion" in X.columns
            and "Area_Terreno_Escri" in X.columns
        ):
            X["Potencial_Constructivo"] = (
                X["Pisos_PUGS"] * X["Area_Terreno_Escri"]
            ) - X["Area_Construccion"]
            X["Potencial_Constructivo"] = X["Potencial_Constructivo"].clip(lower=0)
            X["Pct_Potencial_Usado"] = X["Area_Construccion"] / (
                X["Pisos_PUGS"] * X["Area_Terreno_Escri"] + 1
            )
            self.features_creadas.extend(
                ["Potencial_Constructivo", "Pct_Potencial_Usado"]
            )
            count += 2
            print(f"  ✓ Potencial_Constructivo, Pct_Potencial_Usado")

        print(f"  → {count} features creadas")
        return X

    def get_feature_names(self) -> List[str]:
        """Retorna lista de features creadas"""
        return self.features_creadas

    def get_feature_info(self) -> Dict[str, List[str]]:
        """Retorna información organizada de las features creadas"""
        info = {
            "areas": [
                f
                for f in self.features_creadas
                if any(
                    x in f
                    for x in ["Area", "Ratio_Construccion", "Frente", "Profundidad"]
                )
            ],
            "geoespaciales": [
                f
                for f in self.features_creadas
                if any(
                    x in f
                    for x in ["Distancia", "Lat", "Lon", "Cuadrante", "Norte", "Este"]
                )
            ],
            "influencias": [f for f in self.features_creadas if "Influencia" in f],
            "temporales": [
                f
                for f in self.features_creadas
                if any(x in f for x in ["Edad", "Nuevo", "Moderno", "Viejo", "Decada"])
            ],
            "regulacion": [
                f
                for f in self.features_creadas
                if any(x in f for x in ["Cos", "Cus", "Potencial", "Margen"])
            ],
        }
        return info


def aplicar_feature_engineering(
    X: pd.DataFrame,
    centro_lat: float = -0.18294996618702658,
    centro_lon: float = -78.48456375891507,
    anio_actual: int = 2025,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Función wrapper para aplicar feature engineering de forma simple

    Args:
        X: DataFrame con features originales
        centro_lat: Latitud del centro de referencia
        centro_lon: Longitud del centro de referencia
        anio_actual: Año actual para cálculos

    Returns:
        Tuple con (DataFrame transformado, lista de features creadas)
    """
    engineer = RealEstateFeatureEngineer(centro_lat, centro_lon, anio_actual)
    X_transformed = engineer.fit_transform(X)
    return X_transformed, engineer.get_feature_names()
