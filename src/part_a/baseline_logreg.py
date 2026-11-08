import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import time

# 1. Загружаем подготовленные данные
print("Загрузка данных...")
train = pd.read_csv('../../data/train_prepared.csv')
test = pd.read_csv('../../data/test_prepared.csv')

# Отделяем признаки (X) от цели (y)
X_train = train.drop(columns=['y'])
y_train = train['y']
X_test = test.drop(columns=['y'])
y_test = test['y']

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# 2. Создаем модель
# class_weight='balanced' — это тот самый трюк, чтобы модель не игнорировала редкие 'yes'
model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)

# 3. Обучаем и замеряем время
print("Обучение модели...")
start_time = time.time()
model.fit(X_train, y_train)
train_time = time.time() - start_time

# 4. Предсказываем на тесте
start_time = time.time()
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1] # Вероятность того, что скажут 'yes'
predict_time = time.time() - start_time

# 5. Считаем метрики
print("\n=== РЕЗУЛЬТАТЫ ЛОГИСТИЧЕСКОЙ РЕГРЕССИИ ===")
print(f"Accuracy (Точность): {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision (Точность 'yes'): {precision_score(y_test, y_pred):.4f}")
print(f"Recall (Полнота 'yes'): {recall_score(y_test, y_pred):.4f}")
print(f"F1-мера: {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
print(f"AP (Average Precision): {average_precision_score(y_test, y_proba):.4f}")

print(f"\nВремя обучения: {train_time:.4f} сек")
print(f"Время предсказания: {predict_time:.4f} сек")