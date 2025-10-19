import warnings
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier, XGBRegressor

    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor

    LIGHTGBM_AVAILABLE = True
except:
    LIGHTGBM_AVAILABLE = False


class RegressionModels:
    """Entrenamiento y comparación de modelos de regresión"""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None

    def get_models(self) -> Dict:
        """Retorna diccionario de modelos a entrenar"""
        models = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(random_state=self.random_state),
            "Lasso": Lasso(random_state=self.random_state),
            "ElasticNet": ElasticNet(random_state=self.random_state),
            "RandomForest": RandomForestRegressor(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            ),
            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=100, random_state=self.random_state
            ),
        }

        if XGBOOST_AVAILABLE:
            models["XGBoost"] = XGBRegressor(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            )

        if LIGHTGBM_AVAILABLE:
            models["LightGBM"] = LGBMRegressor(
                n_estimators=100, random_state=self.random_state, n_jobs=-1, verbose=-1
            )

        return models

    def train_and_evaluate(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        cv: int = 5,
    ) -> Dict:
        """Entrena y evalúa todos los modelos"""
        print("\n" + "=" * 70)
        print("ENTRENANDO MODELOS DE REGRESIÓN")
        print("=" * 70)

        models = self.get_models()

        for name, model in models.items():
            print(
                f"\n[{list(models.keys()).index(name)+1}/{len(models)}] Entrenando {name}..."
            )

            try:
                # Entrenar
                model.fit(X_train, y_train)

                # Predicciones
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)

                # Métricas en train
                r2_train = r2_score(y_train, y_pred_train)
                mae_train = mean_absolute_error(y_train, y_pred_train)
                rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))

                # Métricas en test
                r2_test = r2_score(y_test, y_pred_test)
                mae_test = mean_absolute_error(y_test, y_pred_test)
                rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
                mape_test = mean_absolute_percentage_error(y_test, y_pred_test)

                # Cross-validation
                cv_scores = cross_val_score(
                    model, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1
                )

                # Guardar resultados
                self.results[name] = {
                    "model": model,
                    "r2_train": r2_train,
                    "r2_test": r2_test,
                    "mae_train": mae_train,
                    "mae_test": mae_test,
                    "rmse_train": rmse_train,
                    "rmse_test": rmse_test,
                    "mape_test": mape_test,
                    "cv_mean": cv_scores.mean(),
                    "cv_std": cv_scores.std(),
                }

                print(
                    f"  ✓ R² Test: {r2_test:.4f} | RMSE: {rmse_test:,.2f} | MAE: {mae_test:,.2f}"
                )

            except Exception as e:
                print(f"  ✗ Error entrenando {name}: {str(e)}")
                continue

        # Identificar mejor modelo
        best_name = max(self.results.items(), key=lambda x: x[1]["r2_test"])[0]
        self.best_model = self.results[best_name]["model"]

        print(
            f"\n✓ Mejor modelo: {best_name} (R² = {self.results[best_name]['r2_test']:.4f})"
        )

        return self.results

    def hyperparameter_tuning(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_name: str = "RandomForest",
        search_type: str = "grid",
    ) -> Dict:
        """Optimización de hiperparámetros"""
        print(f"\n{'='*70}")
        print(f"OPTIMIZACIÓN DE HIPERPARÁMETROS - {model_name}")
        print(f"{'='*70}")

        # Grids de hiperparámetros
        param_grids = {
            "RandomForest": {
                "n_estimators": [100, 200, 400, 800],
                "max_depth": [15, 20, 25],
            },
            "GradientBoosting": {
                "n_estimators": [150, 300, 500, 1000],
                "learning_rate": [0.1, 0.15, 0.2],
            },
            "XGBoost": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "subsample": [0.8, 1.0],
            },
        }

        if model_name not in param_grids:
            print(f"⚠ No hay grid definido para {model_name}")
            return {}

        # Obtener modelo base
        base_models = self.get_models()
        if model_name not in base_models:
            print(f"⚠ Modelo {model_name} no disponible")
            return {}

        model = base_models[model_name]
        param_grid = param_grids[model_name]

        # Búsqueda
        if search_type == "grid":
            search = GridSearchCV(
                model, param_grid, cv=3, scoring="r2", n_jobs=-1, verbose=1
            )
        else:
            search = RandomizedSearchCV(
                model,
                param_grid,
                n_iter=20,
                cv=3,
                scoring="r2",
                n_jobs=-1,
                random_state=self.random_state,
                verbose=1,
            )

        search.fit(X_train, y_train)

        results = {
            "best_params": search.best_params_,
            "best_score": search.best_score_,
            "best_model": search.best_estimator_,
        }

        print(f"\n✓ Mejores parámetros encontrados:")
        for param, value in results["best_params"].items():
            print(f"  - {param}: {value}")
        print(f"\n✓ Mejor R² (CV): {results['best_score']:.4f}")

        return results

    def get_feature_importance(
        self, model_name: str = None, top_n: int = 20
    ) -> pd.Series:
        """Obtiene importancia de features"""
        if model_name is None:
            # Usar mejor modelo
            model_name = max(self.results.items(), key=lambda x: x[1]["r2_test"])[0]

        model = self.results[model_name]["model"]

        # Verificar si el modelo tiene feature_importances_
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            return pd.Series(importances).sort_values(ascending=False).head(top_n)
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_)
            return pd.Series(importances).sort_values(ascending=False).head(top_n)
        else:
            return pd.Series()

    def save_models(self, output_dir: str):
        """Guarda todos los modelos entrenados"""
        import os

        os.makedirs(output_dir, exist_ok=True)

        for name, result in self.results.items():
            model_path = f"{output_dir}/{name.lower()}_model.pkl"
            joblib.dump(result["model"], model_path)

        print(f"\n✓ {len(self.results)} modelos guardados en {output_dir}")


class ClassificationModels:
    """Clasificación por rangos de avalúo"""

    def __init__(self, n_classes: int = 5, random_state: int = 42):
        self.n_classes = n_classes
        self.random_state = random_state
        self.models = {}
        self.results = {}

    def create_target_classes(self, y: pd.Series) -> pd.Series:
        """Convierte target continuo a clases"""
        y_classes = pd.qcut(y, q=self.n_classes, labels=False, duplicates="drop")

        print(f"\n✓ Target convertido a {self.n_classes} clases")
        print(f"  Distribución: {y_classes.value_counts().sort_index().to_dict()}")

        return y_classes

    def get_models(self) -> Dict:
        """Modelos de clasificación"""
        models = {
            "LogisticRegression": LogisticRegression(
                max_iter=1000, random_state=self.random_state
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            ),
            "KNN": KNeighborsClassifier(n_neighbors=5),
        }

        if XGBOOST_AVAILABLE:
            models["XGBoost"] = XGBClassifier(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            )

        return models

    def train_and_evaluate(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> Dict:
        """Entrena modelos de clasificación"""
        print("\n" + "=" * 70)
        print("ENTRENANDO MODELOS DE CLASIFICACIÓN")
        print("=" * 70)

        # Convertir target a clases
        y_train_classes = self.create_target_classes(y_train)
        y_test_classes = self.create_target_classes(y_test)

        models = self.get_models()

        for name, model in models.items():
            print(
                f"\n[{list(models.keys()).index(name)+1}/{len(models)}] Entrenando {name}..."
            )

            try:
                model.fit(X_train, y_train_classes)

                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)

                acc_train = accuracy_score(y_train_classes, y_pred_train)
                acc_test = accuracy_score(y_test_classes, y_pred_test)

                self.results[name] = {
                    "model": model,
                    "accuracy_train": acc_train,
                    "accuracy_test": acc_test,
                    "classification_report": classification_report(
                        y_test_classes, y_pred_test
                    ),
                }

                print(f"  ✓ Accuracy Test: {acc_test:.4f}")

            except Exception as e:
                print(f"  ✗ Error: {str(e)}")

        return self.results
