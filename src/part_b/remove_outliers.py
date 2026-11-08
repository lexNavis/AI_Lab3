import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Загружаем агрегированную RFM-таблицу, полученную на предыдущем шаге.
rfm = pd.read_csv('../../results/part_b/rfm.csv', index_col=0)

# Вычисляем порог отсечения для признака Monetary.
# Квантиль 0.995 означает, что 99.5% клиентов имеют сумму покупок ниже этого значения.
# Оставшиеся 0.5% — это аномально крупные оптовики, которые искажают кластеризацию.
threshold = rfm['Monetary'].quantile(0.995)

# Разделяем клиентов на две группы:
# 1. Оптовики (выбросы) — те, чья сумма покупок выше порога.
# 2. Основная масса — все остальные.
wholesale = rfm[rfm['Monetary'] > threshold].copy()
main = rfm[rfm['Monetary'] <= threshold].copy()

print(f"Порог по Monetary: {threshold:.2f}")
print(f"Вынесено оптовиков (выбросов): {len(wholesale)}")
print(f"Осталось клиентов для кластеризации: {len(main)}")

# Сохраняем оптовиков в отдельный файл для дальнейшего анализа и отчета.
wholesale.to_csv('../../results/part_b/wholesale_customers.csv')

# Берем признаки для кластеризации из основной массы.
X = main[['Recency', 'Frequency', 'Monetary', 'UniqueProducts', 'AvgBasket']].copy()

# Применяем логарифмирование. Это нужно, чтобы сгладить сильную асимметрию признаков
# (например, Monetary может отличаться в тысячи раз между клиентами).
X = np.log1p(X)

# Масштабируем признаки (приводим к среднему 0 и стандартному отклонению 1).
# Это критически важно для K-Means и DBSCAN, так как они используют евклидово расстояние.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Сохраняем подготовленные данные, чтобы не пересчитывать их в каждом скрипте кластеризации.
np.save('../../results/part_b/X_scaled.npy', X_scaled)
main.to_csv('../../results/part_b/rfm_main.csv')

print(f"Данные сохранены. Размер X_scaled: {X_scaled.shape}")

# Считаем описательную статистику по очищенной таблице (без оптовиков).
# Используем переменную main, которая уже есть в памяти.
stats = main[['Recency', 'Frequency', 'Monetary', 'UniqueProducts', 'AvgBasket']].describe().T

# Оставляем только нужные колонки.
stats = stats[['min', 'max', 'mean', '50%']]
stats.rename(columns={'50%': 'median'}, inplace=True)

# Добавляем n_missing (пропусков нет) и is_wholesale (оптовиков нет).
stats['n_missing'] = 0
stats['is_wholesale'] = 0

print("\n=== Паспорт агрегированной таблицы (без оптовиков) ===")
print(stats[['n_missing', 'min', 'max', 'mean', 'median']])
