"""
Script para generar el scaler del Experimento A (sin leakage)
Compatible con el modelo RandomForest ya entrenado
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

print("=" * 70)
print("GENERANDO SCALER PARA EXPERIMENTO A (sin leakage)")
print("=" * 70)

# 1. Cargar el modelo para obtener las features exactas
print("\n1️⃣ Cargando modelo del Experimento A...")
modelo_path = Path("output/models/experiment_a/randomforest_model.pkl")

if not modelo_path.exists():
    print(f"❌ ERROR: No se encontró el modelo en: {modelo_path}")
    print("   Ejecuta primero: python main.py")
    exit(1)

modelo = joblib.load(modelo_path)
feature_names = modelo.feature_names_in_

print(f"✅ Modelo cargado")
print(f"   Features esperadas: {len(feature_names)}")
print(f"   Primeras 10: {list(feature_names[:10])}")

# Verificar que NO tiene leakage
if "Aiva_Valor" in feature_names or "Cat_Lote_Id" in feature_names:
    print("❌ ERROR: El modelo tiene features de leakage")
    print("   Asegúrate de usar el modelo del Experimento A")
    exit(1)

print("✅ Modelo correcto (sin leakage)")

# 2. Cargar el dataset de entrenamiento
print("\n2️⃣ Cargando dataset de entrenamiento...")

# Intentar cargar desde varias ubicaciones posibles
posibles_rutas = [
    Path("data/processed/X_train.csv"),
    Path("output/X_train.csv"),
    Path("data/X_train.csv"),
]

X_train = None
for ruta in posibles_rutas:
    if ruta.exists():
        print(f"   Encontrado en: {ruta}")
        X_train = pd.read_csv(ruta)
        break

if X_train is None:
    print("❌ ERROR: No se encontró X_train.csv")
    print("   Ubicaciones buscadas:")
    for ruta in posibles_rutas:
        print(f"   - {ruta}")
    print("\n💡 Solución: Ejecuta python main.py para generar los archivos")
    exit(1)

print(f"✅ Dataset cargado: {X_train.shape}")

# 3. Seleccionar solo las features que usa el modelo
print("\n3️⃣ Seleccionando features del modelo...")

# Verificar que todas las features del modelo existen en X_train
features_faltantes = set(feature_names) - set(X_train.columns)
if features_faltantes:
    print(f"❌ ERROR: Faltan {len(features_faltantes)} features en X_train")
    print(f"   Primeras 10 faltantes: {list(features_faltantes)[:10]}")
    exit(1)

# Seleccionar y reordenar
X_train_modelo = X_train[feature_names]
print(f"✅ Features seleccionadas: {X_train_modelo.shape}")

# 4. Entrenar el scaler
print("\n4️⃣ Entrenando StandardScaler...")
scaler = StandardScaler()
scaler.fit(X_train_modelo)

print(f"✅ Scaler entrenado")
print(f"   Features: {scaler.n_features_in_}")
print(f"   Media de primera feature: {scaler.mean_[0]:.4f}")
print(f"   Std de primera feature: {scaler.scale_[0]:.4f}")

# 5. Guardar el scaler
print("\n5️⃣ Guardando scaler...")

# Guardar en las dos ubicaciones
output_paths = [
    Path("output/models/experiment_a/scaler.pkl"),
    Path("app/scaler.pkl"),
]

for ruta in output_paths:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, ruta)
    print(f"✅ Guardado en: {ruta}")

# 6. Verificación final
print("\n6️⃣ Verificación final...")
scaler_verificado = joblib.load(output_paths[0])

print(f"✅ Scaler cargado correctamente")
print(f"   Features: {scaler_verificado.n_features_in_}")
print(f"   Primeras 10 features: {list(scaler_verificado.feature_names_in_[:10])}")

# Verificar que NO tiene leakage
tiene_aiva = "Aiva_Valor" in scaler_verificado.feature_names_in_
tiene_cat = "Cat_Lote_Id" in scaler_verificado.feature_names_in_

if tiene_aiva or tiene_cat:
    print("❌ ERROR: El scaler tiene features de leakage!")
else:
    print("✅ Scaler SIN leakage - CORRECTO")

print("\n" + "=" * 70)
print("✅ PROCESO COMPLETADO EXITOSAMENTE")
print("=" * 70)
print("\n📁 Archivos generados:")
for ruta in output_paths:
    print(f"   ✅ {ruta}")

print("\n🚀 Siguiente paso:")
print("   streamlit run app/app.py")
print("=" * 70)
