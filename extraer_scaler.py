"""
Script para extraer y guardar el StandardScaler usado en el entrenamiento
Ejecutar este script en el entorno donde se entrenó el modelo

Uso:
    python extraer_scaler.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def extraer_scaler_del_preprocesamiento():
    """
    Extrae el scaler del proceso de preprocesamiento
    Si tienes el pipeline completo, extráelo de ahí
    Si no, crea uno nuevo con el dataset de entrenamiento
    """

    print("=" * 70)
    print("EXTRACCIÓN DEL STANDARDSCALER")
    print("=" * 70)

    # Opción 1: Intentar cargar scaler si ya existe en algún lado
    posibles_rutas = [
        "output/models/scaler.pkl",
        "output/scaler.pkl",
        "scaler.pkl",
    ]

    for ruta in posibles_rutas:
        if Path(ruta).exists():
            print(f"\n✓ Scaler encontrado en: {ruta}")
            scaler = joblib.load(ruta)

            # Guardar en las ubicaciones correctas
            Path("output/models/experiment_a").mkdir(parents=True, exist_ok=True)
            Path("app").mkdir(exist_ok=True)

            joblib.dump(scaler, "output/models/experiment_a/scaler.pkl")
            joblib.dump(scaler, "app/scaler.pkl")

            print("✓ Scaler copiado a:")
            print("  - output/models/experiment_a/scaler.pkl")
            print("  - app/scaler.pkl")
            return scaler

    # Opción 2: Crear scaler nuevo desde el dataset
    print("\n⚠️  Scaler no encontrado, creando uno nuevo...")
    print("    Cargando dataset para entrenar scaler...")

    try:
        # Cargar dataset procesado
        dataset_path = "dataset_final_formateado.xlsx"
        if not Path(dataset_path).exists():
            print(f"\n❌ No se encontró {dataset_path}")
            print("\nOpciones:")
            print("1. Coloca el dataset en la raíz del proyecto")
            print("2. O ejecuta el pipeline completo (main.py) que generará el scaler")
            return None

        df = pd.read_excel(dataset_path)
        print(f"✓ Dataset cargado: {len(df)} registros, {len(df.columns)} columnas")

        # Seleccionar solo columnas numéricas (excluyendo target)
        target_col = "Valoracion_Terreno"

        # Columnas numéricas para escalar
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)

        print(f"✓ Columnas numéricas a escalar: {len(numeric_cols)}")

        # Crear y entrenar scaler
        scaler = StandardScaler()
        scaler.fit(df[numeric_cols])

        print("✓ Scaler entrenado exitosamente")

        # Guardar
        Path("output/models/experiment_a").mkdir(parents=True, exist_ok=True)
        Path("app").mkdir(exist_ok=True)

        joblib.dump(scaler, "output/models/experiment_a/scaler.pkl")
        joblib.dump(scaler, "app/scaler.pkl")

        print("\n✓ Scaler guardado en:")
        print("  - output/models/experiment_a/scaler.pkl")
        print("  - app/scaler.pkl")

        # Verificar
        print("\n📊 Información del Scaler:")
        print(f"  - Mean shape: {scaler.mean_.shape}")
        print(f"  - Scale shape: {scaler.scale_.shape}")
        print(f"  - Features procesadas: {len(scaler.mean_)}")

        return scaler

    except Exception as e:
        print(f"\n❌ Error al crear scaler: {e}")
        print("\n💡 Solución: Ejecuta el pipeline completo (main.py) primero")
        return None


def verificar_scaler(scaler):
    """Verifica que el scaler funcione correctamente"""
    if scaler is None:
        return False

    print("\n" + "=" * 70)
    print("VERIFICACIÓN DEL SCALER")
    print("=" * 70)

    try:
        # Crear datos de prueba
        test_data = np.random.randn(1, len(scaler.mean_))
        scaled = scaler.transform(test_data)

        print(f"✓ Scaler funciona correctamente")
        print(f"  - Input shape: {test_data.shape}")
        print(f"  - Output shape: {scaled.shape}")
        print(f"  - Mean del input: {test_data.mean():.4f}")
        print(f"  - Mean del output: {scaled.mean():.4f}")
        print(f"  - Std del output: {scaled.std():.4f}")

        return True

    except Exception as e:
        print(f"❌ Error al verificar scaler: {e}")
        return False


def main():
    print("\n🔧 Script de Extracción del StandardScaler\n")

    scaler = extraer_scaler_del_preprocesamiento()

    if scaler:
        if verificar_scaler(scaler):
            print("\n" + "=" * 70)
            print("✅ PROCESO COMPLETADO EXITOSAMENTE")
            print("=" * 70)
            print("\nPróximos pasos:")
            print("1. Ejecuta la app: streamlit run app/app.py")
            print("2. La app ahora usará el scaler automáticamente")
            print("3. Las predicciones deberían ser correctas")
        else:
            print("\n" + "=" * 70)
            print("⚠️  SCALER CREADO PERO CON PROBLEMAS")
            print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("❌ NO SE PUDO CREAR EL SCALER")
        print("=" * 70)
        print("\n💡 Solución:")
        print("1. Ejecuta el pipeline completo: python main.py")
        print("2. Esto generará todos los archivos necesarios incluyendo el scaler")
        print("3. Luego ejecuta la app: streamlit run app/app.py")


if __name__ == "__main__":
    main()
