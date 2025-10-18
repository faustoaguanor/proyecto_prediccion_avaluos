
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import json


class ModelEvaluator:
    """Evaluación y comparación de modelos"""
    
    def __init__(self):
        self.comparisons = {}
    
    def compare_experiments(self, results_a: Dict, results_b: Dict,
                          experiment_names: tuple = ('Sin Leakage', 'Con Leakage')) -> pd.DataFrame:
        """Compara resultados de dos experimentos"""
        
        comparison_data = []
        
        for model_name in results_a.keys():
            if model_name in results_b:
                comparison_data.append({
                    'Modelo': model_name,
                    f'R² {experiment_names[0]}': results_a[model_name]['r2_test'],
                    f'R² {experiment_names[1]}': results_b[model_name]['r2_test'],
                    f'RMSE {experiment_names[0]}': results_a[model_name]['rmse_test'],
                    f'RMSE {experiment_names[1]}': results_b[model_name]['rmse_test'],
                    f'MAE {experiment_names[0]}': results_a[model_name]['mae_test'],
                    f'MAE {experiment_names[1]}': results_b[model_name]['mae_test'],
                    'Diferencia R²': results_b[model_name]['r2_test'] - results_a[model_name]['r2_test']
                })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        return df_comparison
    
    def plot_model_comparison(self, results: Dict, metric: str = 'r2_test',
                            save_path: str = None):
        """Grafica comparación de modelos"""
        models = list(results.keys())
        values = [results[m][metric] for m in models]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.barh(models, values, color='steelblue', alpha=0.8)
        
        # Colorear mejor modelo
        max_idx = values.index(max(values))
        bars[max_idx].set_color('darkgreen')
        
        ax.set_xlabel(metric.upper().replace('_', ' '))
        ax.set_title(f'Comparación de Modelos - {metric.upper()}')
        ax.grid(axis='x', alpha=0.3)
        
        # Añadir valores en las barras
        for i, v in enumerate(values):
            ax.text(v, i, f' {v:.4f}', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/model_comparison_{metric}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        return fig
    
    def plot_experiment_comparison(self, df_comparison: pd.DataFrame,
                                  save_path: str = None):
        """Grafica comparación entre experimentos A y B"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # R² comparison
        x = np.arange(len(df_comparison))
        width = 0.35
        
        axes[0].bar(x - width/2, df_comparison.iloc[:, 1], width, 
                   label='Exp A (Sin Leakage)', color='steelblue', alpha=0.8)
        axes[0].bar(x + width/2, df_comparison.iloc[:, 2], width,
                   label='Exp B (Con Leakage)', color='coral', alpha=0.8)
        
        axes[0].set_xlabel('Modelo')
        axes[0].set_ylabel('R² Score')
        axes[0].set_title('Comparación R² - Experimento A vs B')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(df_comparison['Modelo'], rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        # RMSE comparison
        axes[1].bar(x - width/2, df_comparison.iloc[:, 3], width,
                   label='Exp A (Sin Leakage)', color='steelblue', alpha=0.8)
        axes[1].bar(x + width/2, df_comparison.iloc[:, 4], width,
                   label='Exp B (Con Leakage)', color='coral', alpha=0.8)
        
        axes[1].set_xlabel('Modelo')
        axes[1].set_ylabel('RMSE')
        axes[1].set_title('Comparación RMSE - Experimento A vs B')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(df_comparison['Modelo'], rotation=45, ha='right')
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/experiment_comparison.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        return fig
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                      save_path: str = None):
        """Grafica análisis de residuos"""
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.5, s=20)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Predicciones')
        axes[0, 0].set_ylabel('Residuos')
        axes[0, 0].set_title('Residuos vs Predicciones')
        axes[0, 0].grid(alpha=0.3)
        
        # Histogram of residuals
        axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Residuos')
        axes[0, 1].set_ylabel('Frecuencia')
        axes[0, 1].set_title('Distribución de Residuos')
        axes[0, 1].grid(alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot de Residuos')
        
        # Predicted vs Actual
        axes[1, 1].scatter(y_true, y_pred, alpha=0.5, s=20)
        axes[1, 1].plot([y_true.min(), y_true.max()], 
                       [y_true.min(), y_true.max()], 
                       'r--', lw=2)
        axes[1, 1].set_xlabel('Valores Reales')
        axes[1, 1].set_ylabel('Predicciones')
        axes[1, 1].set_title('Predicciones vs Valores Reales')
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/residual_analysis.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        return fig
    
    def generate_summary_report(self, results_a: Dict, results_b: Dict,
                               leakage_report: Dict,
                               output_path: str):
        """Genera reporte HTML completo"""
        
        df_comparison = self.compare_experiments(results_a, results_b)
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Reporte de Análisis Catastral - Target Leakage</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #2c3e50;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #34495e;
                    margin-top: 30px;
                }}
                .warning {{
                    background-color: #fff3cd;
                    border-left: 4px solid #ffc107;
                    padding: 15px;
                    margin: 20px 0;
                }}
                .success {{
                    background-color: #d4edda;
                    border-left: 4px solid #28a745;
                    padding: 15px;
                    margin: 20px 0;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #3498db;
                    color: white;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .metric {{
                    display: inline-block;
                    margin: 10px 20px 10px 0;
                    padding: 15px;
                    background-color: #ecf0f1;
                    border-radius: 5px;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #2c3e50;
                }}
                .metric-label {{
                    font-size: 14px;
                    color: #7f8c8d;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 Reporte de Análisis Catastral con Detección de Target Leakage</h1>
                
                <h2>⚠️ Resumen de Target Leakage Detectado</h2>
                <div class="warning">
                    <strong>Total de columnas sospechosas:</strong> {leakage_report['total_suspicious']}<br>
                    <strong>Detectadas por nombre:</strong> {leakage_report['detection_methods']['by_name']['count']}<br>
                    <strong>Detectadas por estadística:</strong> {leakage_report['detection_methods']['by_correlation']['count']}<br>
                    <br>
                    <strong>Columnas identificadas:</strong><br>
                    {', '.join(leakage_report['all_suspicious_columns'])}
                </div>
                
                <h2>📈 Comparación de Experimentos</h2>
                <p><strong>Experimento A (Sin Leakage):</strong> Modelo entrenado excluyendo features sospechosas</p>
                <p><strong>Experimento B (Con Leakage):</strong> Modelo entrenado incluyendo todas las features</p>
                
                <h3>Tabla Comparativa</h3>
                {df_comparison.to_html(index=False, classes='table')}
                
                <h2>🎯 Métricas del Mejor Modelo (Experimento A)</h2>
                <div class="success">
                    <strong>Recomendación:</strong> Usar el modelo del Experimento A en producción, ya que no incluye features con target leakage.
                    El Experimento B muestra métricas artificialmente infladas debido a la fuga de información.
                </div>
                
                <h2>📋 Conclusiones</h2>
                <ul>
                    <li>El Experimento B (con leakage) muestra métricas superiores debido a la fuga de información del target.</li>
                    <li>Estas métricas NO son representativas del rendimiento real en producción.</li>
                    <li>El Experimento A proporciona una estimación realista del rendimiento esperado.</li>
                    <li>Se recomienda usar el modelo del Experimento A para predicciones en producción.</li>
                </ul>
                
                <p style="margin-top: 40px; color: #7f8c8d; font-size: 12px;">
                    Reporte generado automáticamente por el sistema de análisis catastral.<br>
                    Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
                </p>
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\n✓ Reporte HTML guardado en: {output_path}")

