import pandas as pd
import matplotlib.pyplot as plt

# Загружаем исходные данные
df = pd.read_csv('../../data/bank-additional-full.csv', sep=';')

# Преобразуем целевую переменную в бинарный вид (1 для "yes", 0 для "no")
df['y_bin'] = (df['y'] == 'yes').astype(int)

# Разбиваем данные по времени: первые 80% идут на обучение, последние 20% — на тест.
# Такое разбиение имитирует реальную ситуацию, когда модель обучается на прошлых данных, 
# а проверяется на будущих.
split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx].copy()
test = df.iloc[split_idx:].copy()

# Для оценки стабильности внутри обучающей части делим её на 3 равных последовательных блока.
# Это позволяет увидеть, меняется ли доля положительного класса (дрейф) от начала к концу периода.
n = len(train)
fold_size = n // 3
folds = [
    train.iloc[0:fold_size],
    train.iloc[fold_size:2*fold_size],
    train.iloc[2*fold_size:]
]

# Собираем все данные в один список словарей
drift_data = [
    {"Split": "all", "n": len(df), "n_pos": int(df['y_bin'].sum()), "share_pos": round(df['y_bin'].mean(), 4)},
    {"Split": "train", "n": len(train), "n_pos": int(train['y_bin'].sum()), "share_pos": round(train['y_bin'].mean(), 4)},
    {"Split": "test", "n": len(test), "n_pos": int(test['y_bin'].sum()), "share_pos": round(test['y_bin'].mean(), 4)},
    {"Split": "fold0_val", "n": len(folds[0]), "n_pos": int(folds[0]['y_bin'].sum()), "share_pos": round(folds[0]['y_bin'].mean(), 4)},
    {"Split": "fold1_val", "n": len(folds[1]), "n_pos": int(folds[1]['y_bin'].sum()), "share_pos": round(folds[1]['y_bin'].mean(), 4)},
    {"Split": "fold2_val", "n": len(folds[2]), "n_pos": int(folds[2]['y_bin'].sum()), "share_pos": round(folds[2]['y_bin'].mean(), 4)},
]

# Создаем DataFrame и сохраняем
drift_df = pd.DataFrame(drift_data)
drift_df.to_csv('../../results/part_a/drift_share_pos.csv', index=False)
print("Таблица дрейфа сохранена в ../../results/part_a/drift_share_pos.csv")
print(drift_df)

# --- Построение графика дрейфа ---
# Собираем данные для графика: доли положительного класса в трёх блоках train и в тесте.
labels = ['Блок 0\n(train)', 'Блок 1\n(train)', 'Блок 2\n(train)', 'Тест\n(hold-out)']
shares = [
    folds[0]['y_bin'].mean(),
    folds[1]['y_bin'].mean(),
    folds[2]['y_bin'].mean(),
    test['y_bin'].mean()
]

# Строим линейный график, чтобы наглядно показать, как меняется доля во времени.
plt.figure(figsize=(8, 4))
plt.plot(labels, shares, marker='o', linestyle='-', color='b', markersize=8)
plt.title('Изменение доли положительного класса по последовательным частям выборки')
plt.ylabel('Доля положительного класса')
plt.xlabel('Последовательные части выборки')
plt.grid(True, linestyle='--', alpha=0.7)

# Подписываем каждую точку значением в процентах, чтобы цифры были видны на графике.
for i, v in enumerate(shares):
    plt.text(i, v + 0.01, f'{v:.1%}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('../../results/figures/drift_share_pos.png')
print("\nГрафик сохранен в ../../results/figures/drift_share_pos.png")

# Считаем скользящее среднее (rolling mean) с окном 411 записей.
# Это сглаживает случайные колебания и показывает общий тренд доли отклика.
window = 411
rolling_share = df['y_bin'].rolling(window=window).mean()

# Индекс границы train/test (80% данных)
split_idx = int(len(df) * 0.8)

# Строим график
plt.figure(figsize=(12, 5))
plt.plot(rolling_share, label=f'rolling share y=1 (window={window})')

# Рисуем вертикальную пунктирную линию на границе train/test
plt.axvline(x=split_idx, color='black', linestyle='--', label='train/test boundary')

# Настраиваем подписи и сохраняем
plt.title('Дрейф доли отклика во времени')
plt.xlabel('index (время наблюдения)')
plt.ylabel('доля y=1')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('../../results/figures/drift_target_over_time.png')
print("График дрейфа сохранен в ../../results/figures/drift_target_over_time.png")

