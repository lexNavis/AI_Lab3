import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                             roc_auc_score, average_precision_score, 
                             precision_recall_curve, roc_curve, auc)
import time

# ==========================================
# 1. Загрузка данных и предсказаний
# ==========================================
print("Загрузка данных...")
train = pd.read_csv('../../data/train_prepared.csv')
test = pd.read_csv('../../data/test_prepared.csv')

X_train = train.drop(columns=['y']).astype(float).values
y_train = train['y'].astype(float).values
X_test = test.drop(columns=['y']).astype(float).values
y_test = test['y'].astype(float).values

# Добавляем bias (колонку единиц)
X_train_bias = np.column_stack([np.ones(X_train.shape[0]), X_train])
X_test_bias = np.column_stack([np.ones(X_test.shape[0]), X_test])

# Загружаем предсказания собственной модели 
# Путь: из src/part_a/ в results/part_a/
p_own = pd.read_csv('../../results/part_a/logreg_preds_own.csv', header=None).values.flatten()

# ==========================================
# 2. Обучение библиотечной модели (sklearn)
# ==========================================
print("Обучение sklearn...")
start_time = time.time()
sklearn_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
sklearn_model.fit(X_train_bias, y_train)
sklearn_time = time.time() - start_time

p_lib = sklearn_model.predict_proba(X_test_bias)[:, 1]

# ==========================================
# 3. Подбор порога и расчет метрик
# ==========================================
def evaluate(y_true, y_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    y_pred = (y_proba >= best_threshold).astype(int)
    
    return {
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred),
        'ROC-AUC': roc_auc_score(y_true, y_proba),
        'AP': average_precision_score(y_true, y_proba),
        'Threshold': best_threshold
    }

metrics_own = evaluate(y_test, p_own)
metrics_lib = evaluate(y_test, p_lib)

# ==========================================
# 4. Вывод таблицы сравнения
# ==========================================
print("\n=== Сравнение собственной и библиотечной (sklearn) реализаций ===")
print(f"{'Метрика':<12} | {'Своя (NumPy)':<13} | {'Библиотечная':<13} | {'Разница':<10}")
print("-" * 55)
for metric in ['Precision', 'Recall', 'F1', 'ROC-AUC', 'AP']:
    diff = metrics_own[metric] - metrics_lib[metric]
    print(f"{metric:<12} | {metrics_own[metric]:.4f}        | {metrics_lib[metric]:.4f}        | {diff:+.4f}")
print(f"{'Порог':<12} | {metrics_own['Threshold']:.4f}        | {metrics_lib['Threshold']:.4f}")

print(f"\nВремя обучения (sklearn): {sklearn_time:.4f} сек")

# Сохраняем таблицу в CSV
results_df = pd.DataFrame([
    {'Модель': 'Своя (NumPy)', 'Precision': metrics_own['Precision'], 'Recall': metrics_own['Recall'], 
     'F1': metrics_own['F1'], 'ROC-AUC': metrics_own['ROC-AUC'], 'AP': metrics_own['AP'], 'Порог': metrics_own['Threshold']},
    {'Модель': 'Библиотечная (sklearn)', 'Precision': metrics_lib['Precision'], 'Recall': metrics_lib['Recall'], 
     'F1': metrics_lib['F1'], 'ROC-AUC': metrics_lib['ROC-AUC'], 'AP': metrics_lib['AP'], 'Порог': metrics_lib['Threshold']}
])
# Сохраняем в results/part_a/
results_df.to_csv('../../results/part_a/own_vs_sklearn_logreg.csv', index=False)
print("\nТаблица сохранена в results/part_a/own_vs_sklearn_logreg.csv")

# ==========================================
# 5. Построение ROC и PR кривых (Рисунок 11)
# ==========================================
# ROC
fpr, tpr, _ = roc_curve(y_test, p_own)
roc_auc_val = auc(fpr, tpr)

# PR
precision, recall, _ = precision_recall_curve(y_test, p_own)
ap_val = average_precision_score(y_test, p_own)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# График ROC
axes[0].plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc_val:.3f})')
axes[0].plot([0, 1], [0, 1], color='gray', linestyle='--')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC-кривая (собственная LogReg)')
axes[0].legend(loc="lower right")

# График PR
axes[1].plot(recall, precision, color='green', lw=2, label=f'PR curve (AP = {ap_val:.3f})')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('PR-кривая (собственная LogReg)')
axes[1].legend(loc="upper right")

plt.tight_layout()
# Сохраняем в results/figures/
plt.savefig('../../results/figures/roc_pr_own.png')
print("\nROC и PR кривые сохранены в results/figures/roc_pr_own.png")