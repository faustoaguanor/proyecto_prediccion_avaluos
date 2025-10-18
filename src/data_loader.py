import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class DataLoader:
    """Carga y normalización inicial del dataset catastral - ACTUALIZADO PARA EXCEL"""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = None
        self.metadata = {}

    def load_data(self) -> pd.DataFrame:
        """
        Carga el archivo (CSV o EXCEL) con manejo robusto.
        Detecta automáticamente el formato por la extensión.
        """
        file_ext = self.filepath.lower().split(".")[-1]

        if file_ext in ["xlsx", "xls"]:
            # Cargar archivo Excel
            print(f"✓ Detectado archivo Excel: {self.filepath}")
            try:
                self.df = pd.read_excel(
                    self.filepath, engine="openpyxl" if file_ext == "xlsx" else "xlrd"
                )
                print(f"✓ Archivo Excel cargado exitosamente")
                self.metadata["file_type"] = "excel"
                self.metadata["encoding"] = "N/A (Excel)"
            except Exception as e:
                raise ValueError(f"Error al cargar Excel: {e}")

        elif file_ext == "csv":
            # Cargar archivo CSV (código anterior)
            print(f"✓ Detectado archivo CSV: {self.filepath}")
            encodings = ["utf-8", "latin1", "iso-8859-1", "cp1252"]

            for encoding in encodings:
                try:
                    # Intentar detectar separador
                    sample = pd.read_csv(
                        self.filepath,
                        nrows=5,
                        encoding=encoding,
                        sep=None,
                        engine="python",
                    )

                    # Detectar separador más probable
                    if ";" in str(sample.columns):
                        sep = ";"
                    elif "," in str(sample.columns):
                        sep = ","
                    else:
                        sep = ";"  # Default

                    # Cargar completo
                    self.df = pd.read_csv(
                        self.filepath, sep=sep, encoding=encoding, low_memory=False
                    )
                    print(
                        f"✓ Archivo CSV cargado exitosamente con encoding: {encoding}"
                    )
                    self.metadata["encoding"] = encoding
                    self.metadata["separator"] = sep
                    self.metadata["file_type"] = "csv"
                    break
                except (UnicodeDecodeError, Exception):
                    continue

            if self.df is None:
                raise ValueError(
                    "No se pudo cargar el archivo CSV con ningún encoding probado"
                )

        else:
            raise ValueError(
                f"Formato de archivo no soportado: {file_ext}. Use .xlsx, .xls o .csv"
            )

        self.metadata["shape_original"] = self.df.shape
        self.metadata["columns_original"] = list(self.df.columns)

        return self.df

    def normalize_column_names(self) -> pd.DataFrame:
        """Normaliza nombres de columnas a snake_case"""
        self.df.columns = (
            self.df.columns.str.strip()
            .str.replace(" ", "_")
            .str.replace("[^a-zA-Z0-9_]", "", regex=True)
        )

        self.metadata["columns_normalized"] = list(self.df.columns)
        print(f"✓ Columnas normalizadas: {len(self.df.columns)} columnas")

        return self.df

    def convert_numeric_columns(self) -> pd.DataFrame:
        """Convierte columnas numéricas detectando formato con coma decimal si es CSV"""

        # Solo aplicar conversión de coma decimal si es CSV con separador ;
        if (
            self.metadata.get("file_type") == "csv"
            and self.metadata.get("separator") == ";"
        ):
            for col in self.df.columns:
                if self.df[col].dtype == "object":
                    # Intentar conversión numérica
                    sample = self.df[col].dropna().astype(str).head(100)

                    # Detectar patrón numérico con coma
                    if sample.str.contains(
                        r"^\d+,\d+$|^\d+\.\d+,\d+$", regex=True
                    ).any():
                        try:
                            self.df[col] = (
                                self.df[col]
                                .astype(str)
                                .str.replace(".", "", regex=False)
                                .str.replace(",", ".", regex=False)
                                .replace("nan", np.nan)
                            )
                            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
                        except:
                            pass

        print(f"✓ Conversión numérica completada")
        return self.df

    def parse_date_columns(self) -> pd.DataFrame:
        """Detecta y parsea columnas de fecha/año"""
        date_keywords = ["anio", "ano", "fecha", "date", "year"]

        for col in self.df.columns:
            if any(kw in col.lower() for kw in date_keywords):
                try:
                    if self.df[col].dtype == "object":
                        self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

                    # Si parece un año (4 dígitos entre 1900-2100)
                    if self.df[col].between(1900, 2100).sum() > len(self.df) * 0.5:
                        self.metadata[f"{col}_type"] = "year"
                except:
                    pass

        return self.df

    def basic_cleaning(self) -> pd.DataFrame:
        """Limpieza básica: valores vacíos comunes"""
        empty_values = [
            "",
            " ",
            "NO TIENE",
            "NO RELEVADA",
            "SIN DATO",
            "N/A",
            "NA",
            "NULL",
            "NONE",
            "No Tiene",
        ]

        for col in self.df.columns:
            if self.df[col].dtype == "object":
                self.df[col] = self.df[col].replace(empty_values, np.nan)

        print(f"✓ Limpieza básica completada")
        return self.df

    def remove_id_columns(self) -> pd.DataFrame:
        """
        Elimina columnas de ID que no deben usarse como features.
        En el nuevo dataset: Cat_Lote_Id es la clave primaria.
        """
        id_patterns = ["_id", "id_", "codigo", "cat_lote_id", "numero_predio"]

        cols_to_remove = []
        for col in self.df.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in id_patterns):
                cols_to_remove.append(col)

        if cols_to_remove:
            print(f"\n⚠️  Eliminando columnas de ID (no son features predictivas):")
            for col in cols_to_remove:
                print(f"  - {col}")
            # self.df = self.df.drop(columns=cols_to_remove)
            self.metadata["removed_id_columns"] = cols_to_remove

        return self.df

    def get_summary(self) -> Dict:
        """Resumen del dataset cargado"""
        summary = {
            "shape": self.df.shape,
            "missing_percentage": (
                self.df.isnull().sum() / len(self.df) * 100
            ).to_dict(),
            "dtypes": self.df.dtypes.astype(str).to_dict(),
            "numeric_columns": self.df.select_dtypes(
                include=[np.number]
            ).columns.tolist(),
            "categorical_columns": self.df.select_dtypes(
                include=["object"]
            ).columns.tolist(),
        }

        return summary

    def run_pipeline(self) -> Tuple[pd.DataFrame, Dict]:
        """Ejecuta todo el pipeline de carga"""
        print("\n" + "=" * 70)
        print("INICIANDO CARGA Y NORMALIZACIÓN DE DATOS")
        print("=" * 70)

        self.load_data()
        self.normalize_column_names()
        self.convert_numeric_columns()
        self.parse_date_columns()
        self.basic_cleaning()
        self.remove_id_columns()  # ✅ NUEVO: Elimina Cat_Lote_Id

        summary = self.get_summary()

        print(f"\n✓ Pipeline de carga completado")
        print(f"  - Filas: {self.df.shape[0]:,}")
        print(f"  - Columnas: {self.df.shape[1]}")
        print(f"  - Columnas numéricas: {len(summary['numeric_columns'])}")
        print(f"  - Columnas categóricas: {len(summary['categorical_columns'])}")

        return self.df, self.metadata
