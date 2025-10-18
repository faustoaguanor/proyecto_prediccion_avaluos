import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class EDA:
    """Análisis exploratorio de datos"""
    
    def __init__(self, df: pd.DataFrame, target_col: str):
        self.df = df
        self.target_col = target_col
        self.figures = {}
        
        # Configuración de estilo
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10
    
    def univariate_analysis(self, save_path: str = None):
        """Análisis univariado del target"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Histograma
        axes[0, 0].hist(self.df[self.target_col].dropna(), bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_title(f'Distribución de {self.target_col}')
        axes[0, 0].set_xlabel('Valor')
        axes[0, 0].set_ylabel('Frecuencia')
        
        # Boxplot
        axes[0, 1].boxplot(self.df[self.target_col].dropna())
        axes[0, 1].set_title(f'Boxplot de {self.target_col}')
        axes[0, 1].set_ylabel('Valor')
        
        # QQ-plot
        from scipy import stats
        stats.probplot(self.df[self.target_col].dropna(), dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        
        # Estadísticas
        desc_stats = self.df[self.target_col].describe()
        axes[1, 1].axis('off')
        stats_text = f"""
        Estadísticas Descriptivas:
        
        Count:    {desc_stats['count']:,.0f}
        Mean:     ${desc_stats['mean']:,.2f}
        Std:      ${desc_stats['std']:,.2f}
        Min:      ${desc_stats['min']:,.2f}
        25%:      ${desc_stats['25%']:,.2f}
        Median:   ${desc_stats['50%']:,.2f}
        75%:      ${desc_stats['75%']:,.2f}
        Max:      ${desc_stats['max']:,.2f}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                       verticalalignment='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/univariate_analysis.png", dpi=300, bbox_inches='tight')
        
        self.figures['univariate'] = fig
        plt.close()
        
        return fig
    
    def correlation_analysis(self, top_n: int = 20, save_path: str = None):
        """Análisis de correlación con el target"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != self.target_col and c != 'is_outlier']
        
        correlations = {}
        for col in numeric_cols:
            try:
                corr = self.df[col].corr(self.df[self.target_col])
                correlations[col] = corr
            except:
                pass
        
        # Top correlaciones
        corr_series = pd.Series(correlations).sort_values(ascending=False)
        top_positive = corr_series.head(top_n)
        top_negative = corr_series.tail(top_n)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Positivas
        top_positive.plot(kind='barh', ax=axes[0], color='steelblue')
        axes[0].set_title(f'Top {top_n} Correlaciones Positivas con {self.target_col}')
        axes[0].set_xlabel('Correlación')
        axes[0].grid(axis='x', alpha=0.3)
        
        # Negativas
        top_negative.plot(kind='barh', ax=axes[1], color='coral')
        axes[1].set_title(f'Top {top_n} Correlaciones Negativas con {self.target_col}')
        axes[1].set_xlabel('Correlación')
        axes[1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/correlation_analysis.png", dpi=300, bbox_inches='tight')
        
        self.figures['correlation'] = fig
        plt.close()
        
        return fig, corr_series
    
    def feature_distributions(self, top_features: List[str], save_path: str = None):
        """Distribución de las features más importantes"""
        n_features = min(len(top_features), 9)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for idx, col in enumerate(top_features[:n_features]):
            if col in self.df.columns:
                self.df[col].hist(bins=30, ax=axes[idx], edgecolor='black', alpha=0.7)
                axes[idx].set_title(f'{col}')
                axes[idx].set_xlabel('Valor')
                axes[idx].set_ylabel('Frecuencia')
        
        # Ocultar ejes vacíos
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/feature_distributions.png", dpi=300, bbox_inches='tight')
        
        self.figures['distributions'] = fig
        plt.close()
        
        return fig
    
    def scatter_top_correlations(self, top_n: int = 6, save_path: str = None):
        """Scatter plots de las features con mayor correlación"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != self.target_col and c != 'is_outlier']
        
        correlations = {}
        for col in numeric_cols:
            try:
                corr = abs(self.df[col].corr(self.df[self.target_col]))
                correlations[col] = corr
            except:
                pass
        
        top_cols = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        n_cols = 3
        n_rows = (top_n + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for idx, (col, corr) in enumerate(top_cols):
            axes[idx].scatter(self.df[col], self.df[self.target_col], alpha=0.3, s=10)
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel(self.target_col)
            axes[idx].set_title(f'{col}\n(corr: {corr:.3f})')
            axes[idx].grid(alpha=0.3)
        
        # Ocultar ejes vacíos
        for idx in range(len(top_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/scatter_correlations.png", dpi=300, bbox_inches='tight')
        
        self.figures['scatter'] = fig
        plt.close()
        
        return fig
    
    def run_full_eda(self, save_path: str = None) -> Dict:
        """Ejecuta EDA completo"""
        print("\n" + "="*70)
        print("EJECUTANDO ANÁLISIS EXPLORATORIO (EDA)")
        print("="*70)
        
        results = {}
        
        print("\n[1/4] Análisis univariado del target...")
        self.univariate_analysis(save_path)
        
        print("[2/4] Análisis de correlaciones...")
        _, corr_series = self.correlation_analysis(save_path=save_path)
        results['correlations'] = corr_series.to_dict()
        
        print("[3/4] Distribución de features principales...")
        top_features = corr_series.abs().sort_values(ascending=False).head(9).index.tolist()
        self.feature_distributions(top_features, save_path)
        
        print("[4/4] Scatter plots de correlaciones principales...")
        self.scatter_top_correlations(save_path=save_path)
        
        print("\n✓ EDA completado")
        
        return results

