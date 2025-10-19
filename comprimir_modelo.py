# comprimir_modelo.py
from pathlib import Path

import joblib

print("🗜️ Comprimiendo modelo RandomForest...")

# Cargar modelo original
modelo_path = Path("output/models/experiment_a/randomforest_model.pkl")

if not modelo_path.exists():
    print(f"❌ No se encontró: {modelo_path}")
    exit(1)

modelo = joblib.load(modelo_path)
print(f"✅ Modelo cargado: {modelo.n_features_in_} features")

# Guardar comprimido (compress=3 es el nivel óptimo)
modelo_comprimido_path = Path(
    "output/models/experiment_a/randomforest_model_compressed.pkl"
)
joblib.dump(modelo, modelo_comprimido_path, compress=3)

# Comparar tamaños
import os

tamaño_original = os.path.getsize(modelo_path) / (1024 * 1024)
tamaño_comprimido = os.path.getsize(modelo_comprimido_path) / (1024 * 1024)
reduccion = (1 - tamaño_comprimido / tamaño_original) * 100

print(f"\n📊 RESULTADOS:")
print(f"   Original:    {tamaño_original:.1f} MB")
print(f"   Comprimido:  {tamaño_comprimido:.1f} MB")
print(f"   Reducción:   {reduccion:.1f}%")
print(f"\n✅ Guardado en: {modelo_comprimido_path}")
