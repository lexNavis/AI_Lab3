import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Загружаем данные
rfm = pd.read_csv('../../results/part_b/rfm.csv', index_col=0)
X = rfm[['Recency', 'Frequency', 'Monetary', 'UniqueProducts', 'AvgBasket']].copy()
X = np.log1p(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Считаем расстояния до 5-го соседа
min_samples = 5
neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)

# Берем расстояние до 5-го соседа (последний столбец) и сортируем
distances = np.sort(distances[:, min_samples - 1], axis=0)

# Строим график
plt.figure(figsize=(8, 4))
plt.plot(distances)
plt.title('K-distance graph (min_samples=5)')
plt.xlabel('Points sorted by distance')
plt.ylabel('Distance to 5th nearest neighbor')
plt.grid(True)
plt.savefig('../../results/figures/kdistance_graph.png')
print("График сохранен в results/figures/kdistance_graph.png")

# Выводим значения для анализа
print("Минимальное расстояние:", distances.min())
print("Максимальное расстояние:", distances.max())
print("Медиана:", np.median(distances))
print("90-й перцентиль:", np.percentile(distances, 90))
