import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Загружаем исходный датасет. В этом файле разделитель — точка с запятой.
df = pd.read_csv('../../data/bank-additional-full.csv', sep=';')

# Статистику считаем только на обучающей части (первые 80% по времени).
# Тестовую часть не трогаем, чтобы не было утечки данных (data leakage).
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx].copy()

print("=== 1. ОПИСАТЕЛЬНАЯ СТАТИСТИКА (только train) ===")

# Выбираем только числовые колонки (исключаем текстовые категории).
numeric_cols = train_df.select_dtypes(include=[np.number]).columns

# Временная копия данных, где маркер pdays=999 заменен на NaN.
# Это нужно только для расчета корректных средних и квартилей, 
# чтобы «999» (что значит "не звонили") не искажал статистику реальных дней.
train_df_stats = train_df.copy()
train_df_stats.loc[train_df_stats['pdays'] == 999, 'pdays'] = np.nan

# Считаем статистику по очищенной копии.
stats = train_df_stats[numeric_cols].describe().T

# Добавляем пропуски, дисперсию, асимметрию и эксцесс (требуется в задании).
stats['n_missing'] = train_df_stats[numeric_cols].isnull().sum()
stats['var'] = train_df_stats[numeric_cols].var()
stats['skew'] = train_df_stats[numeric_cols].skew()
stats['kurtosis'] = train_df_stats[numeric_cols].kurtosis()

# Переименовываем колонки для отчета (q1, median, q3) и считаем межквартильный размах (IQR).
stats = stats.rename(columns={'25%': 'q1', '50%': 'median', '75%': 'q3', 'count': 'n'})
stats['IQR'] = stats['q3'] - stats['q1']

# Выбираем нужные колонки в правильном порядке для сохранения.
final_stats = stats[['n', 'n_missing', 'min', 'max', 'mean', 'median', 'var', 'std', 'q1', 'q3', 'IQR', 'skew', 'kurtosis']]
final_stats.to_csv('../../results/part_a/numeric_stats.csv')

print(final_stats[['mean', 'median', 'std', 'skew']])
print("\nПолная таблица статистики сохранена в results/part_a/numeric_stats.csv")

print("\n=== 2. КОРРЕЛЯЦИИ (Пирсон и Спирмен) ===")
# Считаем линейную корреляцию (Пирсон) и монотонную (Спирмен).
corr_pearson = train_df[numeric_cols].corr(method='pearson')
corr_spearman = train_df[numeric_cols].corr(method='spearman')

# Выводим топ-3 самых сильных связей (по модулю) для Пирсона.
print("\nСамые сильные связи (Пирсон):")
print(corr_pearson.abs().unstack().sort_values(ascending=False).drop_duplicates().head(4))

print("\n=== 3. ГИСТОГРАММЫ КОЛИЧЕСТВЕННЫХ ПРИЗНАКОВ ===")
cols_to_plot = [c for c in numeric_cols if c != 'y']

n_cols = 3
n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
axes = axes.flatten()

for i, col in enumerate(cols_to_plot):
    if col == 'pdays':
        # Для pdays строим график только по тем, у кого был прошлый контакт (исключаем 999).
        # Иначе маркер 999 доминирует и скрывает реальное распределение дней.
        data_to_plot = train_df[train_df['pdays'] != 999]['pdays']
        sns.histplot(data_to_plot, bins=30, kde=False, ax=axes[i])
        axes[i].set_title('pdays (без 999)')
    elif col in ['campaign', 'previous']:
        # Для сильно скошенных признаков используем логарифмическую шкалу по X, чтобы хвосты были видны.
        sns.histplot(train_df[col], bins=30, kde=False, log_scale=(True, False), ax=axes[i])
        axes[i].set_title(f'{col} (log X)')
    else:
        sns.histplot(train_df[col], bins=30, kde=False, ax=axes[i])
        axes[i].set_title(f'{col}')
    axes[i].set_xlabel('')

# Убираем пустые оси, если признаки не заполнили всю сетку.
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig('../../results/figures/histograms_all.png')
print("Гистограммы сохранены в results/figures/histograms_all.png")

print("\n=== 4. РАСПРЕДЕЛЕНИЕ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ ===")
# Оцениваем дисбаланс классов в обучающей части.
print(train_df['y'].value_counts(normalize=True) * 100)

print("\n=== 5. КАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ ===")

# Выбираем текстовые колонки и считаем частоты/доли для каждой категории.
cat_cols = train_df.select_dtypes(include=['object']).columns
cat_stats = []

for col in cat_cols:
    counts = train_df[col].value_counts()
    percents = train_df[col].value_counts(normalize=True) * 100
    n_categories = train_df[col].nunique()
    
    print(f"\nПризнак '{col}': {n_categories} категорий")
    
    for cat in counts.index:
        print(f"  {cat}: {counts[cat]} ({percents[cat]:.2f}%)")
        cat_stats.append({
            'Признак': col,
            'Категория': cat,
            'Частота': counts[cat],
            'Доля (%)': round(percents[cat], 2)
        })

# Сохраняем таблицу с частотами категорий.
cat_stats_df = pd.DataFrame(cat_stats)
cat_stats_df.to_csv('../../results/part_a/categorical_stats.csv', index=False)
print("\nСтатистика категориальных признаков сохранена в results/part_a/categorical_stats.csv")

# Строим столбчатую диаграмму распределения целевой переменной.
plt.figure(figsize=(6, 4))
sns.countplot(data=train_df, x='y')
plt.title('Распределение целевой переменной (train)')
plt.savefig('../../results/figures/target_dist.png')
print("График целевой переменной сохранен в results/figures/target_dist.png")

# Создаем бинарную целевую переменную (1 для "yes", 0 для "no") для расчета корреляций.
train_df['y_bin'] = (train_df['y'] == 'yes').astype(int)

# Добавляем бинарную цель к числовым колонкам.
numeric_cols_with_y = list(numeric_cols) + ['y_bin']

# Считаем корреляции.
corr_pearson = train_df[numeric_cols_with_y].corr(method='pearson')
corr_spearman = train_df[numeric_cols_with_y].corr(method='spearman')

# Строим две тепловые карты (Пирсон и Спирмен) рядом.
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(corr_pearson, ax=axes[0], cmap='coolwarm', center=0, vmin=-1, vmax=1)
axes[0].set_title('Корреляция Пирсона')
sns.heatmap(corr_spearman, ax=axes[1], cmap='coolwarm', center=0, vmin=-1, vmax=1)
axes[1].set_title('Корреляция Спирмена')
plt.tight_layout()
plt.savefig('../../results/figures/correlations.png')
print("Тепловые карты сохранены в results/figures/correlations.png")

# --- Зависимость признаков от класса (боксплоты для всех числовых признаков) ---
# Берем все числовые колонки, кроме y_bin и duration (duration исключен как утечка).
features = [c for c in numeric_cols if c not in ['y_bin', 'duration']]

n_cols = 3
n_rows = (len(features) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 12))
axes = axes.flatten()

for i, col in enumerate(features):
    if col == 'pdays':
        # Для pdays снова исключаем маркер 999.
        data = train_df[train_df['pdays'] != 999]
        sns.boxplot(data=data, x='y', y='pdays', ax=axes[i])
        axes[i].set_title('pdays (без 999)')
    elif col in ['campaign', 'previous']:
        # Для скошенных признаков используем логарифмическую шкалу Y, чтобы ящики не слипались.
        sns.boxplot(data=train_df, x='y', y=col, ax=axes[i])
        axes[i].set_yscale('symlog')
        axes[i].set_title(f'{col} (log Y)')
    else:
        sns.boxplot(data=train_df, x='y', y=col, ax=axes[i])
        axes[i].set_title(f'{col}')
    axes[i].set_xlabel('Отклик (y)')
    axes[i].set_ylabel('')

# Убираем пустые оси.
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig('../../results/figures/target_by_class.png')
print("Боксплоты для всех признаков сохранены в results/figures/target_by_class.png")