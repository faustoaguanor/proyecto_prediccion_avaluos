import os
import sys

sys.path.append("..")


def test_imports():
    """Verificar que todos los módulos se importen correctamente"""
    print("Probando imports...")

    try:
        from src.clustering import ClusteringAnalysis
        from src.data_loader import DataLoader
        from src.eda import EDA
        from src.evaluate import ModelEvaluator
        from src.feature_engineering import FeatureEngineer
        from src.models import ClassificationModels, RegressionModels
        from src.preprocessing import LeakageDetector, Preprocessor

        print("✓ Todos los imports exitosos")
        return True
    except Exception as e:
        print(f"✗ Error en imports: {e}")
        return False


def test_leakage_detector():
    """Verificar detección de leakage"""
    print("\\nProbando detector de leakage...")

    from src.preprocessing import LeakageDetector

    detector = LeakageDetector("valoracion_total")

    # Test detección por nombre
    test_columns = [
        "codigo",
        "superficie",
        "valoracion_construccion",  # Debería detectarse
        "valoracion_terreno",  # Debería detectarse
        "avaluo_adicional",  # Debería detectarse
        "zona",
        "precio_promedio",  # Debería detectarse
    ]

    suspicious = detector.detect_by_name(test_columns)

    expected = [
        "valoracion_construccion",
        "valoracion_terreno",
        "avaluo_adicional",
        "precio_promedio",
    ]

    if set(suspicious) == set(expected):
        print(f"✓ Detección correcta: {len(suspicious)} columnas sospechosas")
        return True
    else:
        print(f"✗ Detección incorrecta")
        print(f"  Esperado: {expected}")
        print(f"  Obtenido: {suspicious}")
        return False


def run_all_tests():
    """Ejecutar todos los tests"""
    print("=" * 70)
    print("EJECUTANDO TESTS DEL PROYECTO")
    print("=" * 70)

    tests = [test_imports, test_leakage_detector]

    results = []
    for test in tests:
        results.append(test())

    print("\\n" + "=" * 70)
    if all(results):
        print("✓ TODOS LOS TESTS PASARON")
    else:
        print("✗ ALGUNOS TESTS FALLARON")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
