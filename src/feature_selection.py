"""
Feature Selection para reducir dimensionalidad y mejorar modelos
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression


class FeatureSelector:
    """
    Selecciona las mejores features para evitar overfitting
    """

    def __init__(self, max_features=None, correlation_threshold=0.95):
        """
        Args:
            max_features: Número máximo de features a mantener (None = no límite)
            correlation_threshold: Umbral para eliminar features correlacionadas
        """
        self.max_features = max_features
        self.correlation_threshold = correlation_threshold
        self.selected_features = None
        self.feature_importance_scores = None
        self.removed_features = []

    def eliminar_correlacionadas(self, X, verbose=True):
        """
        Elimina features altamente correlacionadas
        """
        if verbose:
            print("\n" + "=" * 70)
            print("ELIMINANDO FEATURES CORRELACIONADAS")
            print("=" * 70)

        # Calcular matriz de correlación
        corr_matrix = X.corr().abs()

        # Encontrar pares correlacionados
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Features a eliminar
        to_drop = []
        correlaciones_altas = []

        for column in upper_triangle.columns:
            correlated = upper_triangle[column][
                upper_triangle[column] > self.correlation_threshold
            ]
            if len(correlated) > 0:
                for idx in correlated.index:
                    correlaciones_altas.append(
                        {
                            "feature1": column,
                            "feature2": idx,
                            "correlacion": upper_triangle.loc[idx, column],
                        }
                    )
                    to_drop.append(column)

        to_drop = list(set(to_drop))

        if verbose and len(to_drop) > 0:
            print(
                f"\n⚠ Encontradas {len(correlaciones_altas)} correlaciones > {self.correlation_threshold}"
            )
            print(f"\nEjemplos de features correlacionadas:")
            for i, corr in enumerate(correlaciones_altas[:5]):
                print(
                    f"  {i+1}. {corr['feature1']} <-> {corr['feature2']}: {corr['correlacion']:.3f}"
                )
            if len(correlaciones_altas) > 5:
                print(f"  ... y {len(correlaciones_altas)-5} más")

            print(f"\n✂ Eliminando {len(to_drop)} features correlacionadas:")
            for feat in to_drop[:10]:
                print(f"  - {feat}")
            if len(to_drop) > 10:
                print(f"  ... y {len(to_drop)-10} más")

        self.removed_features.extend(to_drop)
        X_reduced = X.drop(columns=to_drop)

        if verbose:
            print(f"\n✓ Features: {X.shape[1]} → {X_reduced.shape[1]}")

        return X_reduced

    def calcular_importancia(self, X, y, method="random_forest", verbose=True):
        """
        Calcula importancia de features

        Args:
            method: 'random_forest' o 'mutual_info'
        """
        if verbose:
            print("\n" + "=" * 70)
            print(f"CALCULANDO IMPORTANCIA ({method.upper()})")
            print("=" * 70)

        if method == "random_forest":
            # Usar Random Forest
            rf = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            )
            rf.fit(X, y)
            importances = pd.Series(rf.feature_importances_, index=X.columns)

        elif method == "mutual_info":
            # Usar Mutual Information
            mi_scores = mutual_info_regression(X, y, random_state=42)
            importances = pd.Series(mi_scores, index=X.columns)

        else:
            raise ValueError("method debe ser 'random_forest' o 'mutual_info'")

        # Ordenar
        importances = importances.sort_values(ascending=False)
        self.feature_importance_scores = importances

        if verbose:
            print(f"\n📊 Top 10 features más importantes:")
            for i, (feat, score) in enumerate(importances.head(10).items(), 1):
                print(f"  {i:2d}. {feat:40s} {score:.4f}")

        return importances

    def seleccionar_mejores(self, X, y, method="random_forest", verbose=True):
        """
        Selecciona las mejores features basándose en importancia
        """
        # Eliminar correlacionadas primero
        X_reduced = self.eliminar_correlacionadas(X, verbose=verbose)

        # Calcular importancia
        importances = self.calcular_importancia(
            X_reduced, y, method=method, verbose=verbose
        )

        # Seleccionar top features
        if self.max_features is not None:
            n_features = min(self.max_features, len(importances))
            selected = importances.head(n_features).index.tolist()

            if verbose:
                print("\n" + "=" * 70)
                print(f"SELECCIÓN FINAL: TOP {n_features} FEATURES")
                print("=" * 70)
                print(f"✓ Features seleccionadas: {len(selected)}")
                print(f"✗ Features eliminadas: {X.shape[1] - len(selected)}")
        else:
            selected = importances.index.tolist()

        self.selected_features = selected
        return X_reduced[selected]

    def transform(self, X):
        """
        Aplica la selección de features a nuevos datos
        """
        if self.selected_features is None:
            raise ValueError("Debes ejecutar fit_transform primero")

        # Eliminar features removidas por correlación
        X_clean = X.drop(columns=[f for f in self.removed_features if f in X.columns])

        # Seleccionar features importantes
        return X_clean[self.selected_features]

    def plot_importances(self, top_n=20, figsize=(12, 8)):
        """
        Grafica las importancias de features
        """
        if self.feature_importance_scores is None:
            raise ValueError("Primero debes calcular importancias")

        plt.figure(figsize=figsize)

        # Top N features
        top_features = self.feature_importance_scores.head(top_n)

        # Plot
        ax = top_features.plot(kind="barh", color="steelblue")
        plt.title(
            f"Top {top_n} Features por Importancia", fontsize=14, fontweight="bold"
        )
        plt.xlabel("Importancia", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        plt.gca().invert_yaxis()
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        return plt.gcf()

    def get_feature_stats(self):
        """
        Retorna estadísticas de la selección
        """
        if self.selected_features is None:
            return None

        return {
            "n_features_seleccionadas": len(self.selected_features),
            "n_features_eliminadas_correlacion": len(self.removed_features),
            "features_seleccionadas": self.selected_features,
            "features_eliminadas": self.removed_features,
        }


def seleccionar_features_optimas(
    X_train, y_train, X_val, y_val, max_features=30, correlation_threshold=0.95
):
    """
    Pipeline completo de feature selection

    Args:
        X_train, y_train: Datos de entrenamiento
        X_val, y_val: Datos de validación
        max_features: Número máximo de features a mantener
        correlation_threshold: Umbral para eliminar correlacionadas

    Returns:
        selector: Objeto FeatureSelector entrenado
        X_train_selected: Train con features seleccionadas
        X_val_selected: Validation con features seleccionadas
    """
    print("\n" + "🎯" * 35)
    print("FEATURE SELECTION PIPELINE")
    print("🎯" * 35)

    print(f"\n📋 Configuración:")
    print(f"  - Features originales: {X_train.shape[1]}")
    print(f"  - Max features a mantener: {max_features}")
    print(f"  - Umbral correlación: {correlation_threshold}")

    # Crear selector
    selector = FeatureSelector(
        max_features=max_features, correlation_threshold=correlation_threshold
    )

    # Seleccionar features
    X_train_selected = selector.seleccionar_mejores(
        X_train, y_train, method="random_forest", verbose=True
    )

    # Aplicar a validación
    X_val_selected = selector.transform(X_val)

    # Estadísticas
    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)
    stats = selector.get_feature_stats()
    print(f"  ✓ Features originales:      {X_train.shape[1]}")
    print(f"  ✗ Eliminadas (correlación): {stats['n_features_eliminadas_correlacion']}")
    print(f"  ✓ Features finales:         {stats['n_features_seleccionadas']}")
    print(
        f"  📊 Reducción:               {(1 - stats['n_features_seleccionadas']/X_train.shape[1])*100:.1f}%"
    )

    return selector, X_train_selected, X_val_selected
