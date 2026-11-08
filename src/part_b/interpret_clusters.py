import pandas as pd
import numpy as np

# Загружаем данные с метками K-Means (или Ward, если хочешь его)
rfm = pd.read_csv('../../results/part_b/rfm_clustered.csv', index_col=0)

# Группируем по кластерам и считаем медианы (медиана лучше среднего, так как есть выбросы)
profile = rfm.groupby('Cluster').agg({
    'Recency': 'median',
    'Frequency': 'median',
    'Monetary': 'median',
    'UniqueProducts': 'median',
    'AvgBasket': 'median'
})
profile['Count'] = rfm.groupby('Cluster').size()

print("=== Профили кластеров (медианы) ===")
print(profile)

# Сохраняем в CSV для отчёта
profile.to_csv('../../results/part_b/cluster_profiles.csv')
print("\nСохранено в ../../results/part_b/cluster_profiles.csv")
