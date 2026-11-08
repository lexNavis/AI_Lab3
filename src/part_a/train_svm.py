import pandas as pd
import numpy as np
import time
from sklearn.svm import SVC
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score

# Загрузка данных
train = pd.read_csv('../../data/train_prepared.csv')
test = pd.read_csv('../../data/test_prepared.csv')
X_train = train.drop(columns=['y']).astype(float).values
y_train = train['y'].astype(float).values
X_test = test.drop(columns=['y']).astype(float).values
y_test = test['y'].astype(float).values

# Сетка параметров
param_grid = {
    'C': [0.001, 0.01, 0.1, 1],
    'kernel': ['linear', 'rbf']
}

results = []
config_num = 1

print("=== Запуск SVM (быстрый режим) ===")

for kernel in param_grid['kernel']:
    for C in param_grid['C']:
        print(f"\n[{config_num}/8] Обучение: kernel={kernel}, C={C}...")
        
        start_time = time.time()
        
        # probability=False — КЛЮЧЕВОЙ момент для скорости
        model = SVC(kernel=kernel, C=C, class_weight='balanced', probability=False, random_state=42)
        model.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        
        # Получаем "сырые" оценки
        decision = model.decision_function(X_test)
        
        # Быстрая сигмоида для получения вероятностей (нужно для AP)
        # Масштабируем decision, чтобы сигмоида не была слишком резкой
        y_proba = 1 / (1 + np.exp(-decision))
        y_pred = model.predict(X_test)
        
        f1 = f1_score(y_test, y_pred)
        ap = average_precision_score(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        print(f"    -> F1: {f1:.4f}, AP: {ap:.4f}, ROC-AUC: {roc_auc:.4f}")
        print(f"    -> Время: {train_time:.2f} сек")
        
        results.append({
            "Config": config_num,
            "Kernel": kernel,
            "C": C,
            "F1": round(f1, 4),
            "AP": round(ap, 4),
            "ROC-AUC": round(roc_auc, 4),
            "Time (s)": round(train_time, 2)
        })
        
        config_num += 1

# Сохраняем результаты
results_df = pd.DataFrame(results)
results_df.to_csv('../../results/part_a/svm_fast_results.csv', index=False)
print("\n=== ИТОГОВАЯ ТАБЛИЦА SVM ===")
print(results_df.to_string(index=False))
print("\nСохранено в ../../results/part_a/svm_fast_results.csv")
