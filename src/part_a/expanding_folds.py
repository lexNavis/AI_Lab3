import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

# Загружаем исходные данные
df = pd.read_csv('../../data/bank-additional-full.csv', sep=';')

# Преобразуем целевую переменную в бинарный вид (1 для "yes", 0 для "no").
df['y'] = (df['y'] == 'yes').astype(int)

# Удаляем duration, так как по условию задачи этот признак недоступен до звонка.
df = df.drop(columns=['duration'])

# Разделяем данные по времени: первые 80% идут на обучение, последние 20% — на тест.
split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx].copy()

# Для проведения проверок с расширяющимся окном делим обучающую часть на 4 последовательных блока.
# Каждый блок соответствует определенному временному периоду.
n = len(train)
block_size = n // 4
blocks = [
    train.iloc[0:block_size],
    train.iloc[block_size:2*block_size],
    train.iloc[2*block_size:3*block_size],
    train.iloc[3*block_size:4*block_size]
]

# Выбираем три модели разных семейств для проверки стабильности.
models = {
    'LogReg': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    'DecisionTree': DecisionTreeClassifier(class_weight='balanced', max_depth=5, random_state=42),
    'RandomForest': RandomForestClassifier(class_weight='balanced', n_estimators=50, random_state=42)
}

# Для упрощения и скорости берем только числовые признаки. 
# В финальном пайплайне использовался One-Hot Encoding, но для проверки фолдов этого достаточно.
num_cols = train.select_dtypes(include=[np.number]).columns.drop('y')

results = []

# Проводим 3 последовательные проверки. На каждом шаге обучающая часть расширяется,
# а валидационная — это следующий по времени блок.
for i in range(3):
    # Обучающая выборка: все блоки от 0-го до i-го включительно.
    X_train_fold = pd.concat([blocks[j][num_cols] for j in range(i+1)])
    y_train_fold = pd.concat([blocks[j]['y'] for j in range(i+1)])
    
    # Валидационная выборка: блок i+1 (идет сразу после обучающих).
    X_val_fold = blocks[i+1][num_cols]
    y_val_fold = blocks[i+1]['y']
    
    # Масштабируем признаки. Важно: scaler обучается ТОЛЬКО на тренировочной части фолда, 
    # чтобы не подсматривать статистику валидации.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_fold)
    X_val_scaled = scaler.transform(X_val_fold)
    
    # Обучаем каждую модель и считаем метрику AP (Average Precision).
    for name, model in models.items():
        model.fit(X_train_scaled, y_train_fold)
        y_proba = model.predict_proba(X_val_scaled)[:, 1]
        ap = average_precision_score(y_val_fold, y_proba)
        results.append({'Model': name, 'Fold': i+1, 'AP': round(ap, 4)})

# Собираем результаты в таблицу и сохраняем в CSV.
results_df = pd.DataFrame(results)
pivot = results_df.pivot(index='Model', columns='Fold', values='AP')
print("=== AP на expanding-фолдах ===")
print(pivot)
pivot.to_csv('../../results/part_a/expanding_folds.csv')
print("\nСохранено в results/part_a/expanding_folds.csv")
