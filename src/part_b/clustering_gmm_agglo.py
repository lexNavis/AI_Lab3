import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import time

# Загружаем данные
rfm = pd.read_csv('../../results/part_b/rfm.csv', index_col=0)
X = rfm[['Recency', 'Frequency', 'Monetary', 'UniqueProducts', 'AvgBasket']].copy()
X = np.log1p(X)
X_scaled = StandardScaler().fit_transform(X)

# Словарь для хранения лучших моделей
best_models = {}

print("=== Agglomerative Clustering ===")
for linkage in ['ward', 'average']:
    best_sil = -1
    best_k = None
    best_labels = None
    
    for k in range(2, 11):
        agg = AgglomerativeClustering(n_clusters=k, linkage=linkage)
        labels = agg.fit_predict(X_scaled)
        
        sil = silhouette_score(X_scaled, labels)
        
        # Смотрим на размеры кластеров
        sizes = np.bincount(labels)
        min_size = sizes.min()
        max_size = sizes.max()
        
        print(f"linkage={linkage:>7}, k={k:>2}: Sil={sil:.4f}, min_cluster={min_size}, max_cluster={max_size}")
        
        if sil > best_sil:
            best_sil = sil
            best_k = k
            best_labels = labels
    
    best_models[f'agg_{linkage}'] = (best_k, best_labels, best_sil)
    print(f"--> Лучший для {linkage}: k={best_k}, Sil={best_sil:.4f}\n")

print("\n=== Gaussian Mixture (GMM) ===")
for cov_type in ['full', 'tied']:
    best_sil = -1
    best_k = None
    best_labels = None
    
    for k in range(2, 11):
        gmm = GaussianMixture(n_components=k, covariance_type=cov_type, random_state=42)
        labels = gmm.fit_predict(X_scaled)
        
        sil = silhouette_score(X_scaled, labels)
        
        if sil > best_sil:
            best_sil = sil
            best_k = k
            best_labels = labels
            
        print(f"cov={cov_type:>4}, k={k:>2}: Sil={sil:.4f}")
    
    best_models[f'gmm_{cov_type}'] = (best_k, best_labels, best_sil)
    print(f"--> Лучший для {cov_type}: k={best_k}, Sil={best_sil:.4f}\n")

# Сохраняем лучшие разметки
for name, (k, labels, sil) in best_models.items():
    rfm_out = rfm.copy()
    rfm_out[f'cluster_{name}'] = labels
    rfm_out.to_csv(f'../../results/part_b/rfm_{name}.csv')
    print(f"Сохранено: rfm_{name}.csv (k={k}, Sil={sil:.4f})")
