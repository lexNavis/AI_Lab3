import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# Загружаем данные
rfm = pd.read_csv('../../results/part_b/rfm.csv', index_col=0)
X = rfm[['Recency', 'Frequency', 'Monetary', 'UniqueProducts', 'AvgBasket']].copy()
X = np.log1p(X)
X_scaled = StandardScaler().fit_transform(X)

# Базовые метки (на всех данных, k=2)
base_kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
base_labels = base_kmeans.fit_predict(X_scaled)

print("=== Проверка устойчивости K-Means (k=2) ===")
print("Подвыборка | ARI с базовой разметкой")

aris = []
n_samples = len(X_scaled)

for i in range(5):
    # Берем случайные 80% клиентов
    rng = np.random.default_rng(seed=42 + i)
    idx = rng.choice(n_samples, size=int(n_samples * 0.8), replace=False)
    
    X_sub = X_scaled[idx]
    
    # Обучаем K-Means на подвыборке
    kmeans_sub = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels_sub = kmeans_sub.fit_predict(X_sub)
    
    # Сравниваем с базовой разметкой на тех же объектах
    base_sub = base_labels[idx]
    ari = adjusted_rand_score(base_sub, labels_sub)
    aris.append(ari)
    
    print(f"    {i+1}     | {ari:.4f}")

print(f"\nСредний ARI: {np.mean(aris):.4f}")
print(f"Стандартное отклонение: {np.std(aris):.4f}")
print(f"Минимум: {np.min(aris):.4f}")
print(f"Максимум: {np.max(aris):.4f}")
