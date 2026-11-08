import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# Загружаем данные
rfm = pd.read_csv('../../results/part_b/rfm.csv', index_col=0)
X = rfm[['Recency', 'Frequency', 'Monetary', 'UniqueProducts', 'AvgBasket']].copy()
X = np.log1p(X)
X_scaled = StandardScaler().fit_transform(X)

print("=== Перебор DBSCAN ===")
print("eps | min_samples | Кластеров | Шум (%) | Silhouette")

best_score = -1
best_params = None

for eps in [0.3, 0.5, 0.7, 0.9, 1.2]:
    for min_samples in [3, 5, 10]:
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X_scaled)
        
        # Считаем статистику
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_pct = (labels == -1).mean() * 100
        
        # Silhouette считаем только если кластеров >= 2 и не все шум
        if n_clusters >= 2:
            # Убираем шум для silhouette (иначе он ломается)
            mask = labels != -1
            score = silhouette_score(X_scaled[mask], labels[mask])
        else:
            score = -1
            
        print(f"{eps:.1f} | {min_samples:>11} | {n_clusters:>9} | {noise_pct:>7.1f} | {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_params = (eps, min_samples, n_clusters, noise_pct)

print(f"\nЛучший: eps={best_params[0]}, min_samples={best_params[1]}, "
      f"кластеров={best_params[2]}, шум={best_params[3]:.1f}%, Silhouette={best_score:.4f}")

# Сохраняем лучшую разметку
eps, min_samples = best_params[0], best_params[1]
db = DBSCAN(eps=eps, min_samples=min_samples)
rfm['DBSCAN_Cluster'] = db.fit_predict(X_scaled)
rfm.to_csv('../../results/part_b/rfm_dbscan.csv')
print(f"\nРазметка DBSCAN сохранена в ../../results/part_b/rfm_dbscan.csv")