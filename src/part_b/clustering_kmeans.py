import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Загружаем агрегированные данные (один клиент = одна строка).
# Используем rfm_main.csv, где уже вынесены оптовики.
rfm = pd.read_csv('../../results/part_b/rfm_main.csv', index_col=0)
print(f"Загружено клиентов: {len(rfm)}")

# Берем признаки для кластеризации. 
# CustomerID — это просто идентификатор, его нельзя использовать в расстояниях.
X = rfm[['Recency', 'Frequency', 'Monetary', 'UniqueProducts', 'AvgBasket']].copy()

# Логарифмируем признаки, потому что Monetary и Frequency имеют сильную асимметрию (длинный хвост).
# Без логарифма K-Means будет "видеть" только крупных клиентов, а мелких сгруппирует в один шум.
X = np.log1p(X)  # log1p = log(1+x), чтобы избежать log(0)

# Масштабируем признаки, чтобы все они были в одном диапазоне.
# K-Means использует евклидово расстояние, и без масштабирования Monetary "задавит" Recency.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Данные подготовлены. Размер матрицы признаков:", X_scaled.shape)

# --- Библиотечный K-Means (для сравнения) ---
# Перебираем число кластеров от 2 до 10, чтобы выбрать лучший по силуэту.
print("\nПеребор числа кластеров для K-Means...")
best_k = None
best_score = -1
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"k={k}, Silhouette={score:.4f}")
    if score > best_score:
        best_score = score
        best_k = k

print(f"\nЛучшее k по силуэту: {best_k} (score={best_score:.4f})")

# Обучаем финальную модель с лучшим k.
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=1)
labels = kmeans.fit_predict(X_scaled)

# ==========================================
# СОХРАНЕНИЕ ЛУЧШЕЙ МОДЕЛИ
# ==========================================
joblib.dump(kmeans, '../../results/part_b/best_solution.joblib')
print(f"Модель K-Means (k={best_k}) сохранена в results/part_b/best_solution.joblib")
# ==========================================

# Считаем метрики качества кластеризации.
sil = silhouette_score(X_scaled, labels)
db = davies_bouldin_score(X_scaled, labels)
ch = calinski_harabasz_score(X_scaled, labels)

print(f"\n=== K-Means (k={best_k}) ===")
print(f"Silhouette: {sil:.4f}")
print(f"Davies-Bouldin: {db:.4f}")
print(f"Calinski-Harabasz: {ch:.4f}")

# Сохраняем метки кластеров обратно в таблицу rfm для дальнейшего анализа.
rfm['Cluster'] = labels
rfm.to_csv('../../results/part_b/rfm_clustered.csv')

# Выводим размеры кластеров.
print("\nРазмеры кластеров:")
print(rfm['Cluster'].value_counts().sort_index())

# Строим простой график (первые два признака) для визуальной проверки.
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', s=10)
plt.title(f'K-Means (k={best_k}) — Recency vs Frequency (log-scale)')
plt.xlabel('Recency (scaled)')
plt.ylabel('Frequency (scaled)')
plt.savefig('../../results/figures/kmeans_clusters.png')
print("\nГрафик сохранен в results/figures/kmeans_clusters.png")