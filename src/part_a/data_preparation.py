import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. Загрузка
df = pd.read_csv('../../data/bank-additional-full.csv', sep=';')

# 2. Удаляем duration
df = df.drop(columns=['duration'])

# 3. Удаляем дубликаты
df = df.drop_duplicates()

# 4. ОБРАБОТКА pdays (ПРАВИЛЬНАЯ)
# Создаем флаг: звонили раньше или нет
df['was_contacted_before'] = (df['pdays'] != 999).astype(int)
# Заменяем 999 на -1. Теперь -1 значит "не звонили", а 0, 5, 10... значит "звонили столько-то дней назад"
df['pdays'] = df['pdays'].replace(999, -1)

# 5. Кодирование целевой переменной
df['y'] = (df['y'] == 'yes').astype(int)

# 6. Разделение 80/20 по времени
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

# 7. Выделяем числовые колонки ДО One-Hot Encoding
# (исключаем y, так как её масштабировать не нужно)
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('y')

# 8. Масштабируем числовые признаки на train и применяем к test
scaler = StandardScaler()
train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
test_df[num_cols] = scaler.transform(test_df[num_cols])

# 9. Кодирование категориальных признаков (One-Hot Encoding)
cat_cols = df.select_dtypes(include=['object']).columns
train_df = pd.get_dummies(train_df, columns=cat_cols, drop_first=True)
test_df = pd.get_dummies(test_df, columns=cat_cols, drop_first=True)

# 10. Выравниваем колонки
missing_in_test = set(train_df.columns) - set(test_df.columns)
missing_in_train = set(test_df.columns) - set(train_df.columns)

for col in missing_in_test:
    test_df[col] = 0
for col in missing_in_train:
    train_df[col] = 0

# Упорядочиваем колонки одинаково
test_df = test_df[train_df.columns]

# 9. Сохраняем
train_df.to_csv('../../data/train_prepared.csv', index=False)
test_df.to_csv('../../data/test_prepared.csv', index=False)

print("Данные подготовлены и сохранены (pdays сохранен как -1 для 'не звонили').")
print(f"Итоговое число признаков: {train_df.shape[1] - 1}")