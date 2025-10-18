import pandas as pd
import numpy as np
from typing import Dict
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns


class ClusteringAnalysis:
    """Análisis de clustering por avalúos"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
    
    def kmeans_clustering(self, X: pd.DataFrame, n_clusters: int = 5) -> Dict:
        """Clustering con KMeans"""
        print(f"\n[KMeans] Generando {n_clusters} clusters...")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        
        silhouette = silhouette_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
        
        result = {
            'model': kmeans,
            'labels': labels,
            'n_clusters': n_clusters,
            'silhouette_score': silhouette,
            'davies_bouldin_score': davies_bouldin,
            'inertia': kmeans.inertia_
        }
        
        print(f"  ✓ Silhouette: {silhouette:.3f} | Davies-Bouldin: {davies_bouldin:.3f}")
        
        return result
    
    def dbscan_clustering(self, X: pd.DataFrame, eps: float = 0.5, 
                         min_samples: int = 5) -> Dict:
        """Clustering con DBSCAN"""
        print(f"\n[DBSCAN] eps={eps}, min_samples={min_samples}...")
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        result = {
            'model': dbscan,
            'labels': labels,
            'n_clusters': n_clusters,
            'n_noise': n_noise
        }
        
        # Silhouette solo si hay más de 1 cluster
        if n_clusters > 1:
            mask = labels != -1
            if mask.sum() > 0:
                silhouette = silhouette_score(X[mask], labels[mask])
                result['silhouette_score'] = silhouette
                print(f"  ✓ Clusters: {n_clusters} | Noise: {n_noise} | Silhouette: {silhouette:.3f}")
            else:
                print(f"  ✓ Clusters: {n_clusters} | Noise: {n_noise}")
        else:
            print(f"  ⚠ Solo 1 cluster encontrado, noise: {n_noise}")
        
        return result
    
    def hierarchical_clustering(self, X: pd.DataFrame, 
                              n_clusters: int = 5) -> Dict:
        """Clustering jerárquico"""
        print(f"\n[Agglomerative] Generando {n_clusters} clusters...")
        
        agg = AgglomerativeClustering(n_clusters=n_clusters)
        labels = agg.fit_predict(X)
        
        silhouette = silhouette_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
        
        result = {
            'model': agg,
            'labels': labels,
            'n_clusters': n_clusters,
            'silhouette_score': silhouette,
            'davies_bouldin_score': davies_bouldin
        }
        
        print(f"  ✓ Silhouette: {silhouette:.3f} | Davies-Bouldin: {davies_bouldin:.3f}")
        
        return result
    
    def elbow_method(self, X: pd.DataFrame, max_k: int = 10, 
                    save_path: str = None):
        """Método del codo para determinar k óptimo"""
        print(f"\n[Elbow Method] Probando k de 2 a {max_k}...")
        
        inertias = []
        silhouettes = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(X, labels))
        
        # Graficar
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(k_range, inertias, 'bo-')
        axes[0].set_xlabel('Número de Clusters (k)')
        axes[0].set_ylabel('Inertia')
        axes[0].set_title('Método del Codo')
        axes[0].grid(alpha=0.3)
        
        axes[1].plot(k_range, silhouettes, 'ro-')
        axes[1].set_xlabel('Número de Clusters (k)')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].set_title('Silhouette Score vs k')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/elbow_method.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        return {'k_range': list(k_range), 'inertias': inertias, 'silhouettes': silhouettes}
    
    def run_clustering_analysis(self, X: pd.DataFrame, 
                               n_clusters: int = 5,
                               save_path: str = None) -> Dict:
        """Ejecuta análisis completo de clustering"""
        print("\n" + "="*70)
        print("EJECUTANDO ANÁLISIS DE CLUSTERING")
        print("="*70)
        
        # Método del codo
        print("\n[1/4] Método del codo...")
        elbow_results = self.elbow_method(X, save_path=save_path)
        
        # KMeans
        print("\n[2/4] KMeans clustering...")
        self.results['kmeans'] = self.kmeans_clustering(X, n_clusters)
        
        # DBSCAN
        print("\n[3/4] DBSCAN clustering...")
        self.results['dbscan'] = self.dbscan_clustering(X, eps=0.5, min_samples=5)
        
        # Jerárquico
        print("\n[4/4] Hierarchical clustering...")
        self.results['hierarchical'] = self.hierarchical_clustering(X, n_clusters)
        
        print("\n✓ Clustering completado")
        
        return self.results
