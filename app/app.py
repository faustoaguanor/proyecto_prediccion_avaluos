"""
App de Predicción de Avalúos Catastrales
Modelo: GradientBoosting con 10 features optimizadas
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Avalúos Catastrales",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilos CSS personalizados
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Cargar modelo
@st.cache_resource
def cargar_modelo():
    """Carga el modelo entrenado"""
    try:
        modelo_path = Path(r"app\GradientBoosting.pkl")
        if not modelo_path.exists():
            # Intentar ruta alternativa
            modelo_path = Path(r"app/GradientBoosting.pkl")

        modelo = joblib.load(modelo_path)
        return modelo, None
    except Exception as e:
        return None, str(e)


def validar_inputs(inputs):
    """Valida que los inputs estén en rangos razonables"""
    warnings = []

    # Validaciones básicas
    if inputs["Area_Construccion"] > inputs["Area_Terreno_Escri"]:
        warnings.append(
            "⚠️ El área de construcción no puede ser mayor que el área del terreno"
        )

    if inputs["Ratio_Construccion_Terreno"] > 1:
        warnings.append("⚠️ El ratio construcción/terreno no puede ser mayor que 1")

    if inputs["Pct_Potencial_Usado"] > 1:
        warnings.append(
            "⚠️ El porcentaje de potencial usado no puede ser mayor que 100%"
        )

    return warnings


def crear_gauge_chart(valor, titulo):
    """Crea un gráfico tipo gauge"""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=valor,
            title={"text": titulo},
            gauge={
                "axis": {"range": [None, 1]},
                "bar": {"color": "#1f77b4"},
                "steps": [
                    {"range": [0, 0.3], "color": "#ffebee"},
                    {"range": [0.3, 0.7], "color": "#fff3cd"},
                    {"range": [0.7, 1], "color": "#d4edda"},
                ],
            },
        )
    )
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def main():
    # Header
    st.markdown(
        '<p class="main-header">🏠 Sistema de Predicción de Avalúos Catastrales</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">Modelo GradientBoosting con 10 Features Optimizadas | R² = 0.9477</p>',
        unsafe_allow_html=True,
    )

    # Cargar modelo
    modelo, error = cargar_modelo()

    if error:
        st.error(f"❌ Error al cargar el modelo: {error}")
        st.info(
            "ℹ️ Asegúrate de que el archivo 'GradientBoosting.pkl' esté en la carpeta 'output/models/experiment_a/'"
        )
        return

    # Sidebar - Información del modelo
    with st.sidebar:
        st.image(
            "app/logo.png",
            width="stretch",
        )

        st.markdown("### 📊 Información del Modelo")
        st.markdown(
            """
        - **Algoritmo:** GradientBoosting
        - **R² Score:** 0.9477
        - **RMSE:** $466,085
        - **Features:** 10 optimizadas
        """
        )

        st.markdown("---")
        st.markdown("### 📋 Features Utilizadas")
        features_info = {
            "🏗️ Construcción": ["Area_Construccion", "Area_Por_Piso"],
            "📐 Terreno": ["Area_Terreno_Escri", "Frente_Total"],
            "📍 Ubicación": ["Distancia_Centro"],
            "🏛️ Influencias": ["Infl_Func_Norm", "Infl_Metr_Norm", "Infl_Road_Norm"],
            "📊 Ratios": ["Ratio_Construccion_Terreno", "Pct_Potencial_Usado"],
        }

        for categoria, features in features_info.items():
            st.markdown(f"**{categoria}**")
            for f in features:
                st.markdown(f"  • {f.replace('_', ' ')}")

    # Tabs principales
    tab1, tab2, tab3 = st.tabs(["🎯 Predicción", "📊 Análisis", "ℹ️ Ayuda"])

    with tab1:
        st.markdown("### Ingrese los Datos de la Propiedad")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🏗️ Características de Construcción")

            area_construccion = st.number_input(
                "Área de Construcción (m²)",
                min_value=0.0,
                value=150.0,
                step=10.0,
                help="Área total construida en metros cuadrados",
            )

            area_terreno = st.number_input(
                "Área del Terreno Escriturado (m²)",
                min_value=0.0,
                value=200.0,
                step=10.0,
                help="Área total del terreno según escrituras",
            )

            frente_total = st.number_input(
                "Frente Total (m)",
                min_value=0.0,
                value=10.0,
                step=0.5,
                help="Longitud del frente del terreno",
            )

            area_por_piso = st.number_input(
                "Área por Piso (m²)",
                min_value=0.0,
                value=75.0,
                step=5.0,
                help="Área promedio por piso",
            )

            ratio_construccion = st.slider(
                "Ratio Construcción/Terreno",
                min_value=0.0,
                max_value=1.0,
                value=0.75,
                step=0.05,
                help="Proporción del terreno que está construida",
            )

        with col2:
            st.markdown("#### 📍 Ubicación e Influencias")

            distancia_centro = st.number_input(
                "Distancia al Centro (grados)",
                min_value=0.0,
                value=0.05,
                step=0.01,
                format="%.4f",
                help="Distancia euclidiana al centro de la ciudad",
            )

            infl_func = st.slider(
                "Influencia Funcional (normalizada)",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="Influencia de funcionalidad del sector",
            )

            infl_metr = st.slider(
                "Influencia Metropolitana (normalizada)",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="Influencia de vías metropolitanas",
            )

            infl_road = st.slider(
                "Influencia Vial (normalizada)",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="Influencia de vías principales",
            )

            pct_potencial = st.slider(
                "% Potencial Usado",
                min_value=0.0,
                max_value=1.0,
                value=0.6,
                step=0.05,
                help="Porcentaje del potencial constructivo utilizado",
            )

        # Crear DataFrame con los inputs
        inputs = {
            "Area_Construccion": area_construccion,
            "Frente_Total": frente_total,
            "Area_Terreno_Escri": area_terreno,
            "Infl_Func_Norm": infl_func,
            "Area_Por_Piso": area_por_piso,
            "Distancia_Centro": distancia_centro,
            "Infl_Metr_Norm": infl_metr,
            "Ratio_Construccion_Terreno": ratio_construccion,
            "Infl_Road_Norm": infl_road,
            "Pct_Potencial_Usado": pct_potencial,
        }

        # Validar inputs
        warnings = validar_inputs(inputs)
        if warnings:
            for warning in warnings:
                st.warning(warning)

        st.markdown("---")

        # Botón de predicción
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn2:
            predecir = st.button(
                "🎯 PREDECIR AVALÚO", use_container_width=True, type="primary"
            )

        if predecir:
            # Hacer predicción
            X_pred = pd.DataFrame([inputs])
            prediccion = modelo.predict(X_pred)[0]

            # Mostrar resultado
            st.markdown("---")
            st.markdown("## 💰 Resultado de la Predicción")

            col_r1, col_r2, col_r3 = st.columns(3)

            with col_r1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(
                    label="Avalúo Predicho", value=f"${prediccion:,.2f}", delta=None
                )
                st.markdown("</div>", unsafe_allow_html=True)

            with col_r2:
                # Calcular rango de confianza (±10% como estimación)
                rango_min = prediccion * 0.9
                rango_max = prediccion * 1.1
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(
                    label="Rango Estimado (±10%)",
                    value=f"${rango_min:,.0f} - ${rango_max:,.0f}",
                    delta=None,
                )
                st.markdown("</div>", unsafe_allow_html=True)

            with col_r3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(
                    label="Precio por m²",
                    value=f"${prediccion/area_terreno:,.2f}",
                    delta=None,
                )
                st.markdown("</div>", unsafe_allow_html=True)

            # Información adicional
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown(
                """
            **ℹ️ Información sobre la predicción:**
            - Esta predicción se basa en un modelo con R² = 0.9477 (94.77% de precisión)
            - El RMSE del modelo es de $466,085
            - El rango estimado considera una variabilidad del ±10%
            """
            )
            st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("### 📊 Análisis de Features")

        if "prediccion" in locals():
            # Crear visualización de importancia relativa
            col_a1, col_a2 = st.columns(2)

            with col_a1:
                st.markdown("#### Distribución de Características")

                # Gráfico de ratios
                fig_ratios = go.Figure()
                fig_ratios.add_trace(
                    go.Bar(
                        x=["Construcción/Terreno", "Potencial Usado"],
                        y=[ratio_construccion, pct_potencial],
                        marker_color=["#1f77b4", "#ff7f0e"],
                    )
                )
                fig_ratios.update_layout(
                    title="Ratios de Aprovechamiento",
                    yaxis_title="Proporción",
                    height=300,
                )
                st.plotly_chart(fig_ratios, use_container_width=True)

            with col_a2:
                st.markdown("#### Influencias del Sector")

                # Gráfico radial de influencias
                categories = ["Funcional", "Metropolitana", "Vial"]
                values = [infl_func, infl_metr, infl_road]

                fig_radar = go.Figure()
                fig_radar.add_trace(
                    go.Scatterpolar(
                        r=values, theta=categories, fill="toself", line_color="#1f77b4"
                    )
                )
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=False,
                    height=300,
                )
                st.plotly_chart(fig_radar, use_container_width=True)

            # Tabla de comparación
            st.markdown("#### 📋 Resumen de Inputs")
            df_summary = pd.DataFrame(
                {"Feature": list(inputs.keys()), "Valor": list(inputs.values())}
            )
            st.dataframe(df_summary, use_container_width=True)
        else:
            st.info(
                "👆 Realiza una predicción en la pestaña 'Predicción' para ver el análisis"
            )

    with tab3:
        st.markdown("### ℹ️ Guía de Uso")

        st.markdown(
            """
        #### 📝 Cómo usar la aplicación
        
        1. **Ingresa los datos** de la propiedad en los campos correspondientes
        2. **Valida** que los valores sean coherentes (verifica las advertencias)
        3. **Haz clic** en el botón "PREDECIR AVALÚO"
        4. **Revisa** el resultado y el rango de confianza
        
        #### 🎯 Descripción de Features
        
        **Características de Construcción:**
        - **Área de Construcción:** Superficie total construida en m²
        - **Área del Terreno:** Superficie total del terreno según escrituras
        - **Frente Total:** Longitud del frente que da a la calle
        - **Área por Piso:** Promedio de área por cada piso
        - **Ratio Construcción/Terreno:** Qué proporción del terreno está construida (0-1)
        
        **Ubicación e Influencias:**
        - **Distancia al Centro:** Distancia euclidiana al centro de la ciudad
        - **Influencias Normalizadas:** Valores entre 0 y 1 que representan la cercanía/influencia de:
          - Funcionalidad del sector
          - Vías metropolitanas
          - Vías principales
        - **% Potencial Usado:** Qué porcentaje del potencial constructivo se ha aprovechado
        
        #### ⚠️ Consideraciones
        
        - Los valores deben ser coherentes entre sí
        - El área de construcción no puede exceder el área del terreno
        - Los ratios deben estar entre 0 y 1
        - La predicción es una estimación basada en el modelo entrenado
        """
        )

        st.markdown("---")
        st.markdown(
            """
        <div style='text-align: center; color: #666;'>
        <p>🏛️ <strong>Sistema de Predicción de Avalúos Catastrales</strong></p>
        <p>Modelo GradientBoosting | R² = 0.9477 | RMSE = $466,085</p>
        </div>
        """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
