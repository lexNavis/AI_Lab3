import pandas as pd
import numpy as np
import time

# Модели
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier

# Поиск гиперпараметров и метрики
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score

# 1. Загружаем данные
train = pd.read_csv('../../data/train_prepared.csv')
test = pd.read_csv('../../data/test_prepared.csv')
X_train = train.drop(columns=['y'])
y_train = train['y']
X_test = test.drop(columns=['y'])
y_test = test['y']

# Для сравнения моделей создадим словарь,
# в котором будем перебирать модели и их конфигурации.
# Конфигурация содержит n полей и m значений таких, что m * n = 8
models = {
    "Dummy": {
        # Заглушка (простейшее правило). Не учится вообще, просто всегда предсказывает самый частый класс.
        # Нужен как точка отсчёта: если сложная модель не лучше Dummy — она бесполезна.
        "model": DummyClassifier(strategy='most_frequent'),
        "param_grid": {}
    },
    "Gaussian Naive Bayes": {
        # Наивный байесовский классификатор. Предполагает, что признаки независимы и распределены нормально.
        # var_smoothing — добавляет крошечную дисперсию к признакам, чтобы не было деления на ноль и работали вероятности.
        "model": GaussianNB(),
        "param_grid": {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        }
    },
    "Logreg": {
        # Логистическая регрессия. Строит линейную границу, но сжимает ответ в вероятность.
        # class_weight='balanced' — штрафует сильнее за ошибки на редком классе "yes" (его всего 11%).
        # C — сила регуляризации. Маленькое C (0.001) — сильно штрафует большие веса (проще модель), большое (10000) — доверяет данным.
        "model": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        "param_grid": {
            "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
        }
    },
    "Decision tree": {
        # Дерево решений. Строит иерархию правил вида "если возраст > 30 и кредит = нет, то...".
        # max_depth — максимальная глубина дерева (чтобы не переобучилось).
        # min_samples_split — минимальное число объектов в узле, чтобы его можно было делить дальше.
        "model": DecisionTreeClassifier(class_weight='balanced'),
        "param_grid": {
            'max_depth': [3, 5, 10, None], 
            'min_samples_split': [2, 5]
        }
    },
    "Random forest": {
        # Случайный лес. Много деревьев, каждое обучено на случайной части данных. Итог — голосование.
        # n_estimators — сколько деревьев строить (больше — стабильнее, но дольше).
        # max_depth — ограничение глубины каждого дерева.
        "model": RandomForestClassifier(class_weight='balanced', n_jobs=-1),
        "param_grid": {
            'n_estimators': [50, 100], 
            'max_depth': [5, 10], 
            'min_samples_split': [2, 5]
        }
    },
    "Boosting": {
        # Градиентный бустинг. Строит деревья последовательно: каждое новое исправляет ошибки предыдущих.
        # Обычно даёт лучшую точность на табличных данных.
        # max_iter и learning_rate — количество деревьев и скорость их "обучения".
        # max_depth — ограничение сложности каждого дерева.
        "model": HistGradientBoostingClassifier(),
        "param_grid": {
            'max_iter': [50, 100], 
            'learning_rate': [0.01, 0.1], 
            'max_depth': [3, 5]
        }
    },
    "KNN": {
        # K-ближайших соседей. Ищет похожих клиентов в обучающей выборке и смотрит на их ответы.
        # n_neighbors — сколько соседей учитывать. Мало (3) — чувствителен к шуму, много (21) — сглаживает.
        # weights — как учитывать соседей: uniform (все равны) или distance (ближние важнее).
        "model": KNeighborsClassifier(),
        "param_grid": {
            'n_neighbors': [3, 5, 11, 21],
            'weights': ['uniform', 'distance']
        }
    }
}
# В списке будем хранить метрики каждого метода
results = []

for name, config in models.items():
    model = config["model"]
    param_grid = config["param_grid"]
    
    print(f"Обучаю {name}...")
    start = time.time()
    
    # Определяем время обучения для всех моделей (включая Dummy)
    if name == "Dummy":
        model.fit(X_train, y_train)
        best = model
        best_params = "N/A"
    else:
        grid = GridSearchCV(model, param_grid, cv=3, scoring='average_precision', n_jobs=-1)
        grid.fit(X_train, y_train)
        best = grid.best_estimator_
        best_params = grid.best_params_
    
    train_time = time.time() - start  # теперь время считается для всех
    
    print(f"Лучшие параметры: {best_params}")
    
    start = time.time()
    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:, 1]
    predict_time = time.time() - start
    
    f1 = f1_score(y_test, y_pred)
    ap = average_precision_score(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    print(f"F1: {f1:.4f}, AP: {ap:.4f}, ROC-AUC: {roc_auc:.4f}")
    print(f"Время обучения: {train_time:.2f} сек, Время предсказания: {predict_time:.4f} сек\n")
    
    # Сохраняем в таблицу
    results.append({
        "Model": name,
        "Best Params": str(best_params),
        "F1": f1,
        "AP": ap,
        "ROC-AUC": roc_auc,
        "Train Time (s)": round(train_time, 2),
        "Predict Time (s)": round(predict_time, 4)
    })

# Сохраняем результаты в CSV
results_df = pd.DataFrame(results)
results_df.to_csv('../../results/part_a/models_no_svm.csv', index=False)

print("\n=== ИТОГОВАЯ ТАБЛИЦА ===")
print(results_df.to_string(index=False))
print("\nТаблица сохранена в ../../results/part_a/models_no_svm.csv")
