import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Загружаем агрегированную таблицу (без оптовиков)
rfm = pd.read_csv('../../results/part_b/rfm_main.csv', index_col=0)
features = ['Recency', 'Frequency', 'Monetary', 'UniqueProducts', 'AvgBasket']

# --- 1. Гистограммы (Рисунок 8) ---
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, col in enumerate(features):
    sns.histplot(rfm[col], bins=30, kde=True, ax=axes[i])
    axes[i].set_title(f'{col}')
    axes[i].set_xlabel('')

# Убираем пустую ось
fig.delaxes(axes[-1])

plt.tight_layout()
plt.savefig('../../results/figures/hist_b.png')
print("Гистограммы сохранены в results/figures/hist_b.png")

# --- 2. Тепловые карты корреляций (Рисунок 7) ---
corr_pearson = rfm[features].corr(method='pearson')
corr_spearman = rfm[features].corr(method='spearman')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(corr_pearson, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=axes[0])
axes[0].set_title('Корреляция Пирсона')
sns.heatmap(corr_spearman, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=axes[1])
axes[1].set_title('Корреляция Спирмена')
plt.tight_layout()
plt.savefig('../../results/figures/corr_b.png')
print("Тепловые карты сохранены в results/figures/corr_b.png")

# --- 3. Вывод цифр для текста ---
print("\n=== Корреляции Пирсона ===")
print(corr_pearson)
print("\n=== Корреляции Спирмена ===")
print(corr_spearman)
